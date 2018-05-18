(*
F# port of the original C# example from the CNTK docs:
https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/MNISTClassifier.cs
*)

#load "../../ScriptLoader.fsx"
open CNTK

#r "../../build/CNTK.FSharp.dll"
open CNTK.FSharp
open CNTK.FSharp.Sequential
 
open System.IO

let numClasses = 10
let input = CNTKLib.InputVariable(shape [ 28; 28; 1 ], DataType.Float)
let labels = CNTKLib.InputVariable(shape [ numClasses ], DataType.Float)

let network : Computation =
    Layer.scale (float32 (1./255.))
    |> Layer.add (Conv2D.convolution 
        {    
            Kernel = { Width = 3; Height = 3 } 
            Strides = { Horizontal = 1; Vertical = 1 }
            Filters = 4
            Initializer = Custom(CNTKLib.GlorotUniformInitializer(0.26, -1, 2))
        }
        )
    |> Layer.add Activation.ReLU
    |> Layer.add (Conv2D.pooling
        {
            PoolingType = PoolingType.Max
            Window = { Width = 3; Height = 3 }
            Strides = { Horizontal = 2; Vertical = 2 }
            Padding = true
        }
        )
    |> Layer.add (Conv2D.convolution
        {    
            Kernel ={ Width = 3; Height = 3 } 
            Strides = { Horizontal = 1; Vertical = 1 }
            Filters = 8
            Initializer = Custom(CNTKLib.GlorotUniformInitializer(0.26, -1, 2))
        }
        )
    |> Layer.add Activation.ReLU
    |> Layer.add (Conv2D.pooling
        {
            PoolingType = PoolingType.Max
            Window = { Width = 3; Height = 3 }
            Strides = { Horizontal = 2; Vertical = 2 }
            Padding = true
        }
        )
    |> Layer.add (Layer.dense numClasses)

let spec = {
    Features = input
    Labels = labels
    Model = network
    Loss = CrossEntropyWithSoftmax
    Eval = ClassificationError
    }

// learning

let ImageDataFolder = __SOURCE_DIRECTORY__
let featureStreamName = "features"
let labelsStreamName = "labels"

// set per sample learning rate
let config = {
    MinibatchSize = 64
    Epochs = 5
    Device = DeviceDescriptor.CPUDevice
    Schedule = { Rate = 0.003125; MinibatchSize = 1; Type = SGDLearner }
    }

let source : TextFormatSource = {
    FilePath = Path.Combine(ImageDataFolder, "Train_cntk_text.txt")
    Features = featureStreamName
    Labels = labelsStreamName
    }

let minibatchSource = TextFormat.source (source.Mappings spec)

let trainer = Learner ()
trainer.MinibatchProgress.Add(Minibatch.basicPrint)

let predictor = trainer.learn minibatchSource (featureStreamName,labelsStreamName) config spec
let modelFile = Path.Combine(__SOURCE_DIRECTORY__,"MNISTConvolution.model")

predictor.Save(modelFile)

// validate the model: this still needs a lot of work to look decent

let indexOfLargest xs = 
    let largest = xs |> Seq.max
    xs |> Seq.findIndex ((=) largest) 

let ValidateModelWithMinibatchSource(
    modelFile:string, 
    textSource:TextFormatSource,
    device:DeviceDescriptor
    ) =

        let model = Function.Load(modelFile, device)

        // can this whole block be extracted into a function?
        let imageInput = 
            model.Inputs
            |> Seq.filter (fun i -> i.IsInput)
            |> Seq.exactlyOne

        let labelOutput = model.Output

        let mappings : TextFormat.InputMappings = {
            Features = [ 
                { Variable = imageInput; SourceName = textSource.Features }
                ] 
            Labels = { Variable = labelOutput; SourceName = textSource.Labels }
            }

        let testMinibatchSource = TextFormat.source (textSource.FilePath, mappings)

        let featureStreamInfo = testMinibatchSource.StreamInfo(textSource.Features)
        let labelStreamInfo = testMinibatchSource.StreamInfo(textSource.Labels)

        let batchSize = 50

        let rec countErrors (total,errors) =

            printfn "Total: %i; Errors: %i" total errors

            let minibatchData = testMinibatchSource.GetNextMinibatch((uint32)batchSize, device)

            if (minibatchData = null || minibatchData.Count = 0)
            then (total,errors)        
            else

                let total = total + minibatchData.[featureStreamInfo].numberOfSamples

                // find the index of the largest label value
                let labelData = 
                    minibatchData
                    |> Minibatch.getValues testMinibatchSource textSource.Labels
                    |> Minibatch.getDense labelOutput

                let expectedLabels = 
                    labelData 
                    |> Seq.map indexOfLargest

                // compute the predicted label
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
                
                let outputData = 
                    outputDataMap.[labelOutput]
                    |> Minibatch.getDense labelOutput 
                
                let actualLabels =
                    outputData 
                    |> Seq.map indexOfLargest

                let misMatches = 
                    (actualLabels,expectedLabels)
                    ||> Seq.zip
                    |> Seq.sumBy (fun (a, b) -> if a = b then 0 else 1)

                let errors = errors + misMatches

                if Minibatch.isSweepEnd (minibatchData)
                then (total,errors)
                else countErrors (total,errors)

        countErrors (uint32 0,0)

let testingSource = {
    FilePath = Path.Combine(ImageDataFolder, "Test_cntk_text.txt")
    Features = featureStreamName
    Labels = labelsStreamName
    }

let total,errors = 
    ValidateModelWithMinibatchSource(
        modelFile,
        testingSource,
        DeviceDescriptor.CPUDevice
        )

printfn "Total: %i / Errors: %i" total errors
