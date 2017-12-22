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

let MiniBatchDataIsSweepEnd(minibatchValues:seq<MinibatchData>) =
    minibatchValues 
    |> Seq.exists(fun a -> a.sweepEnd)

// definition / configuration of the network
let imageSize = 28 * 28
let numClasses = 10

let input = CNTKLib.InputVariable(shape [ imageSize ], DataType.Float)
let labels = CNTKLib.InputVariable(shape [ numClasses ], DataType.Float)

let hiddenLayerDim = 200
let scalingFactor = float32 (1./255.)
let classifier = 
    Layer.scale scalingFactor
    |> Layer.stack (Layer.dense hiddenLayerDim)
    |> Layer.stack Activation.sigmoid
    |> Layer.stack (Layer.dense numClasses)

let spec = {
    Features = input
    Labels = labels
    Model = classifier
    Loss = CrossEntropyWithSoftmax
    Eval = ClassificationError
    Schedule = { Rate = 0.003125; MinibatchSize = 1}
    }

// Configuration of the learning data source
let featureStreamName = "features"
let labelsStreamName = "labels"
//let classifierName = "classifierOutput"
let streamConfigurations = 
    ResizeArray<StreamConfiguration>(
        [
            new StreamConfiguration(featureStreamName, imageSize)    
            new StreamConfiguration(labelsStreamName, numClasses)
        ]
        )
let ImageDataFolder = Path.Combine(__SOURCE_DIRECTORY__, "../data/")

let minibatchSource = 
    MinibatchSource.TextFormatMinibatchSource(
        Path.Combine(ImageDataFolder, "Train_cntk_text.txt"), 
        streamConfigurations, 
        MinibatchSource.InfinitelyRepeat)

// learn from the data
let config = {
    MinibatchSize = 64
    Epochs = 5
    Device = DeviceDescriptor.CPUDevice
    }
let learn (source:MinibatchSource) (featureStreamName:string,labelsStreamName:string) (config:Config) (spec:Specification) =

    let (predictor,trainer) = prepare config.Device spec    
    
    let featureStreamInfo = minibatchSource.StreamInfo(featureStreamName)
    let labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName)
    let minibatchSize = uint32 (config.MinibatchSize)
    let device = config.Device

    let rec learnEpoch (step,epoch) = 

        if epoch <= 0
        // we are done : return function
        then predictor
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
            
            // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
            // Batching will not end. Each time minibatchSource completes an sweep (epoch),
            // the last minibatch data will be marked as end of a sweep. We use this flag
            // to count number of epochs.
            let epoch = 
                if (MiniBatchDataIsSweepEnd(minibatchData.Values))
                then epoch - 1
                else epoch

            learnEpoch (step,epoch)

    learnEpoch (0,config.Epochs)

let predictor = learn minibatchSource (featureStreamName,labelsStreamName) config spec
let modelFile = Path.Combine(__SOURCE_DIRECTORY__,"MNISTMLP.model")

predictor.Save(modelFile)

// validate the model

let device = DeviceDescriptor.CPUDevice

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
    device:DeviceDescriptor, 
    maxCount:int
    ) =

        let model : Function = Function.Load(modelFile, device)
        let imageInput = model.Arguments.[0]
        let labelOutput = 
            model.Output
            // |> Seq.filter (fun o -> o.Name = outputName)
            // |> Seq.exactlyOne

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
        device,
        1000)

printfn "Total: %i / Errors: %i" total errors
