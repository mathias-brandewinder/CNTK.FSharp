(*
F# port of the original C# example from the CNTK docs:
https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/MNISTClassifier.cs
*)

// Use the CNTK.fsx file to load the dependencies.

#load "../../CNTK.Sequential.fsx"
open CNTK
open CNTK.Sequential
open System.IO

// definition / configuration of the network

let imageSize = 28 * 28
let numClasses = 10
let input = CNTKLib.InputVariable(shape [ 28; 28; 1 ], DataType.Float)
let labels = CNTKLib.InputVariable(shape [ numClasses ], DataType.Float)

let network : Computation =
    Layer.scale (float32 (1./255.))
    |> Layer.stack (Conv2D.convolution 
        {    
            Kernel = { Width = 3; Height = 3 } 
            OutputFeatures = 4
            Initializer = Custom(CNTKLib.GlorotUniformInitializer(0.26, -1, 2))
        }
        )
    |> Layer.stack Activation.ReLU
    |> Layer.stack (Conv2D.pooling
        {
            PoolingType = PoolingType.Max
            Window = { Width = 3; Height = 3 }
            Stride = { Horizontal = 2; Vertical = 2 }
        }
        )
    |> Layer.stack (Conv2D.convolution
        {    
            Kernel ={ Width = 3; Height = 3 } 
            OutputFeatures = 8
            Initializer = Custom(CNTKLib.GlorotUniformInitializer(0.26, -1, 2))
        }
        )
    |> Layer.stack Activation.ReLU
    |> Layer.stack (Conv2D.pooling
        {
            PoolingType = PoolingType.Max
            Window = { Width = 3; Height = 3 }
            Stride = { Horizontal = 2; Vertical = 2 }
        }
        )
    |> Layer.stack (Layer.dense numClasses)

// alternate version, using sequence:
let network2 : Computation =
    Layer.sequence
        [
            Layer.scale (float32 (1./255.))
            Conv2D.convolution 
                {    
                    Kernel = { Width = 3; Height = 3 } 
                    OutputFeatures = 4
                    Initializer = Custom(CNTKLib.GlorotUniformInitializer(0.26, -1, 2))
                }               
            Activation.ReLU
            Conv2D.pooling
                {
                    PoolingType = PoolingType.Max
                    Window = { Width = 3; Height = 3 }
                    Stride = { Horizontal = 2; Vertical = 2 }
                }
            Conv2D.convolution
                {    
                    Kernel ={ Width = 3; Height = 3 } 
                    OutputFeatures = 8
                    Initializer = Custom(CNTKLib.GlorotUniformInitializer(0.26, -1, 2))
                }
            Activation.ReLU
            Conv2D.pooling
                {
                    PoolingType = PoolingType.Max
                    Window = { Width = 3; Height = 3 }
                    Stride = { Horizontal = 2; Vertical = 2 }
                }
            Layer.dense numClasses
        ]

let spec = {
    Features = input
    Labels = labels
    Model = network2
    Loss = CrossEntropyWithSoftmax
    Eval = ClassificationError
    }

// learning

let ImageDataFolder = __SOURCE_DIRECTORY__
let featureStreamName = "features"
let labelsStreamName = "labels"

let learningSource: DataSource = {
    SourcePath = Path.Combine(ImageDataFolder, "Train_cntk_text.txt")
    Streams = [
        featureStreamName, imageSize
        labelsStreamName, numClasses
        ]
    }

// set per sample learning rate
let config = {
    MinibatchSize = 64
    Epochs = 5
    Device = DeviceDescriptor.CPUDevice
    Schedule = { Rate = 0.003125; MinibatchSize = 1 }
    }
let minibatchSource = textSource learningSource InfinitelyRepeat

let trainer = Learner ()
trainer.MinibatchProgress.Add(basicMinibatchSummary)

let predictor = trainer.learn minibatchSource (featureStreamName,labelsStreamName) config spec
let modelFile = Path.Combine(__SOURCE_DIRECTORY__,"MNISTConvolution.model")

predictor.Save(modelFile)

// validate the model: this still needs a lot of work to look decent

let testingSource: DataSource = {
    SourcePath = Path.Combine(ImageDataFolder, "Test_cntk_text.txt")
    Streams = [
        featureStreamName, imageSize
        labelsStreamName, numClasses
        ]
    }

let testMinibatchSource = textSource testingSource FullDataSweep

let ValidateModelWithMinibatchSource(
    modelFile:string, 
    testMinibatchSource:MinibatchSource,
    featureInputName:string, 
    labelInputName:string, 
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

                if isSweepEnd (minibatchData.Values)
                // if (int total > maxCount)
                then (total,errors)
                else countErrors (total,errors)

        countErrors (uint32 0,0)

let total,errors = 
    ValidateModelWithMinibatchSource(
        modelFile,
        testMinibatchSource,
        featureStreamName, 
        labelsStreamName, 
        DeviceDescriptor.CPUDevice,
        1000)

printfn "Total: %i / Errors: %i" total errors
