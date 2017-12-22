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

// definition / configuration of the network

let imageSize = 28 * 28
let numClasses = 10
let featureStreamName = "features"
let labelsStreamName = "labels"
let input = CNTKLib.InputVariable(shape [ 28; 28; 1 ], DataType.Float, featureStreamName)
let labels = CNTKLib.InputVariable(shape [ numClasses ], DataType.Float, labelsStreamName)
let scalingFactor = float32 (1./255.)
let classifierName = "classifierOutput"
let network  =
    Layer.scale scalingFactor
    |> Layer.stack (Convolution.conv2D 
        {    
            KernelWidth = 3 
            KernelHeight = 3 
            InputChannels = 1
            OutputFeatures = 4
        }
        )
    |> Layer.stack Activation.ReLU
    |> Layer.stack (Convolution.pooling2D
        {
            PoolingType = PoolingType.Max
            WindowWidth = 3
            WindowHeight = 3
            HorizontalStride = 2 
            VerticalStride = 2
        }
        )
    |> Layer.stack (Convolution.conv2D
        {    
            KernelWidth = 3 
            KernelHeight = 3 
            InputChannels = 4 // matches previous conv output
            OutputFeatures = 8
        }
        )
    |> Layer.stack Activation.ReLU
    |> Layer.stack (Convolution.pooling2D
        {
            PoolingType = PoolingType.Max
            WindowWidth = 3
            WindowHeight = 3
            HorizontalStride = 2 
            VerticalStride = 2
        }
        )
    |> Layer.stack (Layer.dense numClasses)

let spec = {
    Features = input
    Labels = labels
    Model = network
    Loss = CrossEntropyWithSoftmax
    Eval = ClassificationError
    Schedule = { Rate = 0.003125; MinibatchSize = 1 }
    }

// learning
let ImageDataFolder = Path.Combine(__SOURCE_DIRECTORY__, "../data/")

let streamConfigurations = 
    ResizeArray<StreamConfiguration>(
        [
            new StreamConfiguration(featureStreamName, imageSize)    
            new StreamConfiguration(labelsStreamName, numClasses)
        ]
        )

let modelFile = Path.Combine(__SOURCE_DIRECTORY__,"MNISTConvolution.model")

let minibatchSource = 
    MinibatchSource.TextFormatMinibatchSource(
        Path.Combine(ImageDataFolder, "Train_cntk_text.txt"), 
        streamConfigurations, 
        MinibatchSource.InfinitelyRepeat)

let featureStreamInfo = minibatchSource.StreamInfo(featureStreamName)
let labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName)

// set per sample learning rate
let config = {
    MinibatchSize = 64
    Epochs = 5
    Device = DeviceDescriptor.CPUDevice
    }
let predictor = learn minibatchSource (featureStreamName,labelsStreamName) config spec

predictor.Save(modelFile)

// validate the model
let minibatchSourceNewModel = 
    MinibatchSource.TextFormatMinibatchSource(
        Path.Combine(ImageDataFolder, "Test_cntk_text.txt"), 
        streamConfigurations, 
        MinibatchSource.FullDataSweep)

let device = DeviceDescriptor.CPUDevice

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
        let labelOutput = model.Output

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
