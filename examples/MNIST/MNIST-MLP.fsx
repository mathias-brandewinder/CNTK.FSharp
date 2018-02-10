(*
F# port of the original C# example from the CNTK docs:
https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/MNISTClassifier.cs
*)

#load "../../ScriptLoader.fsx"
open CNTK

#r "../../build/CNTK.FSharp.dll"
open CNTK.FSharp

open System.IO
open System.Collections.Generic

let FullyConnectedLinearLayer(
    input:Variable, 
    outputDim:int, 
    device:DeviceDescriptor,
    outputName:string) : Function =

    let inputDim = input.Shape.[0]

    let weights = Param.init ([outputDim;inputDim], DataType.Float, GlorotUniform) device

    let timesFunction = weights.Tensor * input.Tensor

    let bias = Param.init ([ outputDim ], DataType.Float, Value 0.0) device
    
    Tensor.plus(bias.Tensor, timesFunction, outputName) 
    |> Tensor.toFunction

let Dense(
    input:Variable, 
    outputDim:int,
    device:DeviceDescriptor,
    outputName:string) : Function =

    input.Tensor 
    |> Tensor.flatten
    |> fun tensor -> 
        FullyConnectedLinearLayer(tensor.toVar, outputDim, device, outputName)
    
// definition / configuration of the network

let ImageDataFolder = __SOURCE_DIRECTORY__

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

let CreateMLPClassifier(
    device:DeviceDescriptor, 
    numOutputClasses:int,
    hiddenLayerDim:int,
    scaledInput:Function,
    classifierName:string) =

        let dense1 = 
            Dense(var scaledInput, hiddenLayerDim, device, "")
            |> (var >> CNTKLib.Sigmoid)
        Dense(var dense1, numOutputClasses, device, classifierName)
        
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

let learn epochs =

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

            if step % outputFrequencyInMinibatches = 0
            then
                trainer
                |> Minibatch.summary 
                |> Minibatch.basicPrint
        
            // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
            // Batching will not end. Each time minibatchSource completes an sweep (epoch),
            // the last minibatch data will be marked as end of a sweep. We use this flag
            // to count number of epochs.
            let epoch = 
                if (Minibatch.isSweepEnd(minibatchData))
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
