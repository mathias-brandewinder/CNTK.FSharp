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

let shape (dims:int seq) = NDShape.CreateNDShape dims

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

let modelFile = Path.Combine(__SOURCE_DIRECTORY__,"MNISTMLP.model")

let input = CNTKLib.InputVariable(shape [ imageSize ], DataType.Float, featureStreamName)
let hiddenLayerDim = 200
let scalingFactor = float32 (1./255.)

let device = DeviceDescriptor.CPUDevice

let scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float32>(scalingFactor, device), input)

let FullyConnectedLinearLayer(
    input:Variable, 
    outputDim:int, 
    device:DeviceDescriptor,
    outputName:string) : Function =

// defaults:
// string outputName = ""
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
        // Variable transform has been inserted
        new Variable(CNTKLib.Times(timesParam, input, "times"))

    let plusParam = new Parameter(shape [ outputDim ], 0.0f, device, "plusParam")
    CNTKLib.Plus(plusParam, timesFunction, outputName)

type Activation = 
    | None
    | ReLU
    | Sigmoid
    | Tanh

let Dense(
    input:Variable, 
    outputDim:int,
    device:DeviceDescriptor,
    activation:Activation, 
    outputName:string) : Function =

    // defaults:
    // Activation activation = Activation.None, string outputName = "")

    let input : Variable =
        if (input.Shape.Rank <> 1)
        then
            let newDim = input.Shape.Dimensions |> Seq.reduce(fun d1 d2 -> d1 * d2)
            // inserted explicit Variable transformation
            new Variable(CNTKLib.Reshape(input, shape [ newDim ]))
        else input

    let fullyConnected : Function = 
        FullyConnectedLinearLayer(input, outputDim, device, outputName)
    
    // inserted explicit Variable transformations
    match activation with
    | Activation.None -> fullyConnected
    | Activation.ReLU -> CNTKLib.ReLU(new Variable(fullyConnected))
    | Activation.Sigmoid -> CNTKLib.Sigmoid(new Variable(fullyConnected))
    | Activation.Tanh -> CNTKLib.Tanh(new Variable(fullyConnected))
    

let CreateMLPClassifier(
    device:DeviceDescriptor, 
    numOutputClasses:int,
    hiddenLayerDim:int,
    scaledInput:Function,
    classifierName:string) =

        // inserted explicit Variable conversions
        let dense1:Function = Dense(new Variable(scaledInput), hiddenLayerDim, device, Activation.Sigmoid, "");
        let classifierOutput:Function = Dense(new Variable(dense1), numOutputClasses, device, Activation.None, classifierName);
        classifierOutput

let classifierOutput = CreateMLPClassifier(device, numClasses, hiddenLayerDim, scaledInput, classifierName)

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

let printTrainingProgress(trainer:Trainer, minibatchIdx:int, outputFrequencyInMinibatches:int) =
        
    if 
        // print out only after every x minibatches
        (minibatchIdx % outputFrequencyInMinibatches) = 0 && 
        trainer.PreviousMinibatchSampleCount() <> (uint32 0)
    then  
        let trainLossValue = trainer.PreviousMinibatchLossAverage() |> float32
        let evaluationValue = trainer.PreviousMinibatchEvaluationAverage() |> float32
        printfn "Minibatch: %i CrossEntropyLoss = %f, EvaluationCriterion = %f" minibatchIdx trainLossValue evaluationValue

let MiniBatchDataIsSweepEnd(minibatchValues:seq<MinibatchData>) =
    minibatchValues 
    |> Seq.exists(fun a -> a.sweepEnd)


// this is terrible, need to rewrite
let mutable i = 0
let mutable epochs = 5

while (epochs > 0) do

    let minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device)
    let arguments : IDictionary<Variable, MinibatchData> =
        [
            input, minibatchData.[featureStreamInfo]
            labels, minibatchData.[labelStreamInfo]
        ]
        |> dict

    trainer.TrainMinibatch(arguments, device)
    i <- i + 1
    printTrainingProgress(trainer, i, outputFrequencyInMinibatches)

    // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
    // Batching will not end. Each time minibatchSource completes an sweep (epoch),
    // the last minibatch data will be marked as end of a sweep. We use this flag
    // to count number of epochs.
    if (MiniBatchDataIsSweepEnd(minibatchData.Values))
    then
        epochs <- epochs - 1



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

// defaults:
// int maxCount = 1000
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

                let inputDataMap : Dictionary<Variable, Value> =
                    let x = Dictionary<Variable, Value>()
                    x.Add(imageInput, minibatchData.[featureStreamInfo].data)
                    x

                let outputDataMap : Dictionary<Variable, Value> =
                    let x = Dictionary<Variable, Value>()
                    x.Add(labelOutput, null)
                    x

                model.Evaluate(inputDataMap, outputDataMap, device)

                let outputData = outputDataMap.[labelOutput].GetDenseData<float32>(labelOutput)
                let actualLabels =
                    outputData 
                    |> Seq.map (fun l ->                         
                        let largest = l |> Seq.max
                        l.IndexOf largest
                        )

                (actualLabels,expectedLabels)
                ||> Seq.zip
                |> Seq.iter (fun (x,y) -> printfn "Real: %A, Pred: %A" x y)

                let misMatches = 
                    (actualLabels,expectedLabels)
                    ||> Seq.zip
                    |> Seq.sumBy (fun (a, b) -> if a = b then 0 else 1)

                let errors = errors + misMatches

                if (int total > maxCount)
                then (total,errors)
                else countErrors (total,errors)

        countErrors (uint32 0,0)

let _ = 
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
