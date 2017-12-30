(*
F# port of the original C# example from the CNTK docs:
https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/LogisticRegression.cs
*)

// Use the CNTK.fsx file to load the dependencies.

#load "../CNTK.Sequential.fsx"
open CNTK
open CNTK.Sequential

open System
open System.Collections.Generic
open System.Text
open System.IO

// Conversion of the original C# code to an F# script

// Helpers to simplify model creation from F#

// Creating a synthetic dataset

let generateGaussianNoise (random:Random) (mean, stdDev) =
        
    let u1 = 1.0 - random.NextDouble()
    let u2 = 1.0 - random.NextDouble()
    let stdNormalRandomValue = sqrt(-2.0 * log(u1)) * sin(2.0 * Math.PI * u2)

    mean + stdDev * stdNormalRandomValue
 
let random = Random(0)

// generate synthetic data: each group is normally distributed,
// centered around (3,3), (6,6), ... 
// this is not as efficient as the original example, but clearer: 
// we create an array of labels first, which we then transform
// into one-hot form, and into its features. 
let generateSyntheticData(sampleSize,inputDim,numOutputClasses) =
        
    // utility one-hot encoder
    let oneHot classes value = 
        Array.init classes (fun i -> 
            if i = value then float32 1. else float32 0.
            )
    
    // generate synthetic feature for given label
    let generateFeatures label =
        Array.init inputDim (fun _ -> 
            generateGaussianNoise random (3.0, 1.0) * (float label + 1.) |> float32
            )

    let labels = Array.init sampleSize (fun _ -> random.Next numOutputClasses)
    let oneHotLabels = labels |> Array.collect (oneHot numOutputClasses)
    let features = labels |> Array.collect (generateFeatures)

    features, oneHotLabels

let minibatchSize = 64
let numMinibatchesToTrain = 1000
let updatePerMinibatches = 50
let inputDim = 3
let numOutputClasses = 2


let builder=new StringBuilder()
[1 .. (numMinibatchesToTrain)]
|>List.iter (fun _ ->
    let str (txt:string) = builder.Append(txt) |> ignore
    let str_ txt = str txt
                   str " "
    let flt value = str_ (value.ToString())
    let features, oneHotLabels = generateSyntheticData(minibatchSize, inputDim, numOutputClasses)
    [0..minibatchSize-1]
    |>List.iter (
          fun batchIndex ->
                let values tag vals =
                    str "|"
                    str_ tag
                    vals |> Array.iter (fun entry ->flt entry)
                values "labels" (Array.sub oneHotLabels (batchIndex*numOutputClasses) numOutputClasses)
                str "\t"
                values "features" (Array.sub features (batchIndex*inputDim) inputDim)
                builder.AppendLine("") |> ignore
                )
    )
let dataFile = Path.Combine(__SOURCE_DIRECTORY__, "logisticData.txt")
File.WriteAllText(dataFile,builder.ToString().Replace(",",".")) |>ignore

let device = DeviceDescriptor.CPUDevice

let featureVariable = Variable.InputVariable(shape[inputDim], DataType.Float)
let labelVariable = Variable.InputVariable(shape[numOutputClasses], DataType.Float)

let network : Computation =
    Layer.scale (float32 (1.))
    |> Layer.stack (Layer.dense numOutputClasses)

let spec = {
    Features = featureVariable
    Labels = labelVariable
    Model = network
    Loss = CrossEntropyWithSoftmax
    Eval = ClassificationError
    }
let featureStreamName = "features"
let labelsStreamName = "labels"

let learningSource: DataSource = {
    SourcePath = dataFile
    Streams = [
              featureStreamName, inputDim
              labelsStreamName, numOutputClasses
        ]
    }
let config = {
    MinibatchSize = 64
    Epochs = 1
    Device = device
    Schedule = { Rate = 0.001; MinibatchSize = 1 }
    }
let outputFrequencyInMinibatches = 20
let minibatchSource = textSource learningSource InfinitelyRepeat
let predictor = learn minibatchSource (featureStreamName,labelsStreamName) config spec
  //                    (fun trainer step -> let report = progress (trainer, outputFrequencyInMinibatches)
  //                                         report step |> printer
  //                    )

let modelFile = Path.Combine(__SOURCE_DIRECTORY__,"logistic.model")
predictor.Save(modelFile)
let streams = 
    learningSource.Streams
    |> Seq.map (fun (name, dim) -> 
        new StreamConfiguration(name, dim))
    |> ResizeArray
// validate the model: this still needs a lot of work to look decent
let minibatchSourceNewModel = 
    MinibatchSource.TextFormatMinibatchSource(
        dataFile, 
        streams, 
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
        [|inputDim|], 
        numOutputClasses, 
        featureStreamName, 
        labelsStreamName, 
        DeviceDescriptor.CPUDevice,
        1000)

printfn "Total: %i / Errors: %i" total errors 