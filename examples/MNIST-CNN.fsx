(*
F# port of the original C# example from the CNTK docs:
https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/MNISTClassifier.cs
*)

// Use the CNTK.fsx file to load the dependencies.

#load "../CNTK.fsx"
open CNTK

open System
open System.IO
open System.Collections.Generic

// Conversion of the original C# code to an F# script

// Helpers to simplify model creation from F#

let FullyConnectedLinearLayer(
    input:Variable, 
    outputDim:int,
    name:string,
    device:DeviceDescriptor) : Function =

    let inputDim = input.Shape.[0]

    let timesParam = 
        new Parameter(
            shape [outputDim; inputDim], 
            DataType.Float,
            CNTKLib.GlorotUniformInitializer(
                float CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, 
                uint32 1),
            device, 
            "timesParam")

    let timesFunction = 
        new Variable(CNTKLib.Times(timesParam, input, "times"))

    let plusParam = new Parameter(shape [ outputDim ], 0.0f, device, "plusParam")
    CNTKLib.Plus(plusParam, timesFunction, name)

let dense
    (device:DeviceDescriptor)
    (outputDim:int)
    // this should not be necessary but setting the name of the function separately 
    // is causing some grief when re-opening the model!?
    (name:string)
    (input:Variable) : Function =

    let input : Variable =
        if (input.Shape.Rank <> 1)
        then
            let newDim = input.Shape.Dimensions |> Seq.reduce(*)
            new Variable(CNTKLib.Reshape(input, shape [ newDim ]))
        else input

    FullyConnectedLinearLayer(input, outputDim, name, device)

// this doesn't seem to behave as I expected!?
let named name (f:Function) = 
    f.SetName name
    f

let Dense(
    input:Variable, 
    outputDim:int,
    device:DeviceDescriptor,
    activation:Activation, 
    outputName:string) : Function =

    let input : Variable =
        if (input.Shape.Rank <> 1)
        then
            let newDim = input.Shape.Dimensions |> Seq.reduce(fun d1 d2 -> d1 * d2)
            new Variable(CNTKLib.Reshape(input, shape [ newDim ]))
        else input

    let fullyConnected : Function = 
        FullyConnectedLinearLayer(input, outputDim, outputName, device)
    
    match activation with
    | Activation.None -> fullyConnected
    | Activation.ReLU -> CNTKLib.ReLU(new Variable(fullyConnected))
    | Activation.Sigmoid -> CNTKLib.Sigmoid(new Variable(fullyConnected))
    | Activation.Tanh -> CNTKLib.Tanh(new Variable(fullyConnected))

let MiniBatchDataIsSweepEnd(minibatchValues:seq<MinibatchData>) =
    minibatchValues 
    |> Seq.exists(fun a -> a.sweepEnd)

// definition / configuration of the network

let ImageDataFolder = Path.Combine(__SOURCE_DIRECTORY__, "../data/")

let featureStreamName = "features"
let labelsStreamName = "labels"
let classifierName = "classifierOutput"
let imageSize = 28 * 28
let numClasses = 10

let streamConfigurations = 
    ResizeArray<StreamConfiguration>(
        [
            new StreamConfiguration(featureStreamName, imageSize)    
            new StreamConfiguration(labelsStreamName, numClasses)
        ]
        )

let modelFile = Path.Combine(__SOURCE_DIRECTORY__,"MNISTConvolution.model")

let input = CNTKLib.InputVariable(shape [ 28; 28; 1 ], DataType.Float, featureStreamName)
let hiddenLayerDim = 200
let scalingFactor = float32 (1./255.)

let device = DeviceDescriptor.CPUDevice

let scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float32>(scalingFactor, device), input)

type Kernel = {
    Width:int
    Height:int
    }

type Conv2D = {
    Kernel:Kernel
    InputChannels:int
    OutputFeatureMap:int
    }

type Window = {
    Width:int
    Height:int 
    }

type Stride = {
    Horizontal:int
    Vertical:int
    }

type Pooling2D = {
    PoolingType:PoolingType
    Window:Window
    Stride:Stride
    }

let pooling2D
    (pooling:Pooling2D)
    (input:Variable) : Function = 

        CNTKLib.Pooling(
            input, 
            pooling.PoolingType,
            shape [ pooling.Window.Width; pooling.Window.Height ], 
            shape [ pooling.Stride.Horizontal; pooling.Stride.Vertical ], 
            [| true |])

let convolution2D   
    (device:DeviceDescriptor)
    (conv:Conv2D)
    (features:Variable) : Function =

        // parameter initialization hyper parameter
        let convWScale = 0.26

        let convParams = 
            new Parameter(
                shape [ 
                    conv.Kernel.Width
                    conv.Kernel.Height
                    conv.InputChannels
                    conv.OutputFeatureMap
                    ], 
                DataType.Float,
                CNTKLib.GlorotUniformInitializer(convWScale, -1, 2), 
                device)

        CNTKLib.Convolution(
            convParams, 
            features, 
            shape [ 1; 1; conv.InputChannels ])

let ReLU = CNTKLib.ReLU

let funcToVar (f:Function) = new Variable(f)

let ConvolutionWithMaxPooling(
    features:Variable, 
    device:DeviceDescriptor,
    kernelWidth:int, 
    kernelHeight:int, 
    numInputChannels:int, 
    outFeatureMapCount:int,
    hStride:int, 
    vStride:int, 
    poolingWindowWidth:int, 
    poolingWindowHeight:int) : Function =

        // parameter initialization hyper parameter
        let convWScale = 0.26

        let convParams = 
            new Parameter(
                shape [ kernelWidth; kernelHeight; numInputChannels; outFeatureMapCount ], 
                DataType.Float,
                CNTKLib.GlorotUniformInitializer(convWScale, -1, 2), 
                device)

        let convFunction : Function = 
            CNTKLib.ReLU(
                new Variable(
                    CNTKLib.Convolution(
                        convParams, 
                        features, 
                        shape [ 1; 1; numInputChannels ])))

        let pooling : Function = 
            CNTKLib.Pooling(
                new Variable(convFunction), 
                PoolingType.Max,
                shape [ poolingWindowWidth; poolingWindowHeight ], 
                shape [ hStride; vStride ], 
                [| true |])

        pooling

let CreateConvolutionalNeuralNetwork(
    features:Variable,
    outDims:int,
    device:DeviceDescriptor, 
    classifierName:string) : Function = 

        // 28x28x1 -> 14x14x4
        features
        |> convolution2D
            device
            { 
                Kernel = { Width = 3; Height = 3 }
                InputChannels = 1
                OutputFeatureMap = 4
            }
        |> funcToVar
        |> ReLU
        |> funcToVar
        |> pooling2D
            {
                PoolingType = PoolingType.Max
                Window = { Width = 3; Height = 3 }
                Stride = { Horizontal = 2; Vertical = 2 }
            }       
        // 14x14x4 -> 7x7x8
        |> funcToVar
        |> convolution2D
            device
            { 
                Kernel = { Width = 3; Height = 3 }
                InputChannels = 4 // same as OutputFeatureMap previous conv
                OutputFeatureMap = 8
            }
        |> funcToVar
        |> ReLU
        |> funcToVar
        |> pooling2D
            {
                PoolingType = PoolingType.Max
                Window = { Width = 3; Height = 3 }
                Stride = { Horizontal = 2; Vertical = 2 }
            }
        // dense final layer
        |> funcToVar
        |> dense device outDims classifierName
        // |> named classifierName


let classifierOutput = CreateConvolutionalNeuralNetwork(new Variable(scaledInput), numClasses, device, classifierName)

let labels = CNTKLib.InputVariable(shape [ numClasses ], DataType.Float, labelsStreamName)
let trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labels, "lossFunction")
let prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labels, "classificationError")

let minibatchSource = 
    MinibatchSource.TextFormatMinibatchSource(
        Path.Combine(ImageDataFolder, "Train_cntk_text.txt"), 
        streamConfigurations, 
        MinibatchSource.InfinitelyRepeat)

let featureStreamInfo = minibatchSource.StreamInfo(featureStreamName)
let labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName)

// set per sample learning rate
let learningRatePerSample : CNTK.TrainingParameterScheduleDouble = 
    new CNTK.TrainingParameterScheduleDouble(0.003125, uint32 1)

let parameterLearners = 
    ResizeArray<Learner>(
        [
            Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample)
        ]
        )

let trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners)

let minibatchSize = uint32 64
let outputFrequencyInMinibatches = 20

let learn epochs =

    let report = progress (trainer, outputFrequencyInMinibatches)

    let rec learnEpoch (step,epoch) = 

        if epoch <= 0
        // we are done
        then ignore ()
        else
            let step = step + 1
            let minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device)

            let arguments : IDictionary<Variable, MinibatchData> =
                [
                    input, minibatchData.[featureStreamInfo]
                    labels, minibatchData.[labelStreamInfo]
                ]
                |> dict

            trainer.TrainMinibatch(arguments, device) |> ignore

            report step |> printer
            
            // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
            // Batching will not end. Each time minibatchSource completes an sweep (epoch),
            // the last minibatch data will be marked as end of a sweep. We use this flag
            // to count number of epochs.
            let epoch = 
                if (MiniBatchDataIsSweepEnd(minibatchData.Values))
                then epoch - 1
                else epoch

            learnEpoch (step,epoch)

    learnEpoch (0,epochs)

let epochs = 5
learn epochs

classifierOutput.Save(modelFile)

// validate the model
let minibatchSourceNewModel = 
    MinibatchSource.TextFormatMinibatchSource(
        Path.Combine(ImageDataFolder, "Test_cntk_text.txt"), 
        streamConfigurations, 
        MinibatchSource.FullDataSweep)

let ValidateModelWithMinibatchSource(
    modelFile:string, 
    testMinibatchSource:MinibatchSource,
    imageDim:int[], 
    numClasses:int, 
    featureInputName:string, 
    labelInputName:string, 
    outputName:string,
    device:DeviceDescriptor, 
    maxCount:int
    ) =

        let model : Function = Function.Load(modelFile, device)
        let imageInput = model.Arguments.[0]
        let labelOutput = 
            model.Outputs 
            |> Seq.filter (fun o -> o.Name = outputName)
            |> Seq.exactlyOne

        let featureStreamInfo = testMinibatchSource.StreamInfo(featureInputName)
        let labelStreamInfo = testMinibatchSource.StreamInfo(labelInputName)

        let batchSize = 50

        let rec countErrors (total,errors) =

            printfn "Total: %i; Errors: %i" total errors

            let minibatchData = testMinibatchSource.GetNextMinibatch((uint32)batchSize, device)

            if (minibatchData = null || minibatchData.Count = 0)
            then (total,errors)        
            else

                let total = total + minibatchData.[featureStreamInfo].numberOfSamples

                // find the index of the largest label value
                let labelData = minibatchData.[labelStreamInfo].data.GetDenseData<float32>(labelOutput)
                let expectedLabels = 
                    labelData 
                    |> Seq.map (fun l ->                         
                        let largest = l |> Seq.max
                        l.IndexOf largest
                        )

                let inputDataMap = 
                    [
                        imageInput, minibatchData.[featureStreamInfo].data
                    ]
                    |> dataMap

                let outputDataMap = 
                    [ 
                        labelOutput, null 
                    ] 
                    |> dataMap
                    
                model.Evaluate(inputDataMap, outputDataMap, device)

                let outputData = outputDataMap.[labelOutput].GetDenseData<float32>(labelOutput)
                let actualLabels =
                    outputData 
                    |> Seq.map (fun l ->                         
                        let largest = l |> Seq.max
                        l.IndexOf largest
                        )

                let misMatches = 
                    (actualLabels,expectedLabels)
                    ||> Seq.zip
                    |> Seq.sumBy (fun (a, b) -> if a = b then 0 else 1)

                let errors = errors + misMatches

                if (int total > maxCount)
                then (total,errors)
                else countErrors (total,errors)

        countErrors (uint32 0,0)

let total,errors = 
    ValidateModelWithMinibatchSource(
        modelFile, 
        minibatchSourceNewModel,
        [|imageSize|], 
        numClasses, 
        featureStreamName, 
        labelsStreamName, 
        classifierName, 
        device,
        1000)

printfn "Total: %i / Errors: %i" total errors
