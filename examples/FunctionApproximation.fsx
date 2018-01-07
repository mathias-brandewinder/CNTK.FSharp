#I @"C:\Users\Andrey\Private\VS_Projects\CNTK.FSharp"

#load "CNTK.Sequential.fsx"
//#load "examples/trainDataParser.fsx"
open CNTK
open CNTK.Sequential


open System
open System.IO
open System.Collections.Generic
open System.Text


let minibatchSize = 200
let numMinibatchesToTrain = 3000
let inputDim = 1
let numOutputClasses = 1
type StorageMode =
|MemoryStorage
|FileStorage
let storageMode = FileStorage

let periods = 2.0f
let fnc_x ind = (float32)ind/(float32)(minibatchSize)
let fnc_y x = float32(sin(x*periods *2.0f * (float32)Math.PI))

let random = Random(0)

let featuresData =  
                [|0..numMinibatchesToTrain-1|] 
                |> Array.map (fun _ -> [|0..minibatchSize-1|] |> Array.map (fun _ -> random.Next (minibatchSize-1)) |> Array.map fnc_x )
                |> Array.concat
let labelsData = featuresData |>  Array.map fnc_y

let device = DeviceDescriptor.CPUDevice
let featureVariable = Variable.InputVariable(shape[inputDim], DataType.Float)
let labelVariable = Variable.InputVariable(shape[numOutputClasses], DataType.Float)

let nNeur = 4
let network : Computation =
    Layer.dense nNeur
    |> Layer.stack Activation.tanh
    |> Layer.stack (Layer.dense nNeur)
    |> Layer.stack Activation.tanh
    |> Layer.stack (Layer.dense nNeur)
    |> Layer.stack Activation.tanh 
    |> Layer.stack (Layer.dense nNeur)
    |> Layer.stack Activation.tanh 
    |> Layer.stack (Layer.dense numOutputClasses)

let spec = {
    Features = featureVariable
    Labels = labelVariable
    Model = network
    Loss = Loss.SquaredError
    Eval = Loss.SquaredError
    }
let featureStreamName = "features"
let labelsStreamName = "labels"

let config = {
    MinibatchSize = minibatchSize
    Epochs = 5
    Device = device
    Schedule = { Rate = 0.0001; MinibatchSize = 1 }
    }

let outputFrequencyInMinibatches = (uint32)(100*minibatchSize)
let coarseMinibatchSummary frequency (summary:TrainingMiniBatchSummary) =
    if (summary.TotalSamples % frequency = (uint32)0) then
        basicMinibatchSummary summary

let learner = Learner ()
learner.MinibatchProgress.Add (coarseMinibatchSummary outputFrequencyInMinibatches)
let sw = System.Diagnostics.Stopwatch()
sw.Start()
let predictor = 
    match storageMode with 
    |MemoryStorage ->
        let dataSize = featuresData |> Seq.length
        let getData step = 
            let startIndex = (step*minibatchSize%dataSize)
            let extractBatch array = Array.sub  array  startIndex minibatchSize
            startIndex + minibatchSize >= featuresData.Length,featuresData |> extractBatch,labelsData |> extractBatch
        let inMemMinibatchHandler = inMemoryMinibatchHandler config spec getData
        learner.learn config spec inMemMinibatchHandler
    |FileStorage ->
        let builder = StringBuilder ()
        featuresData
        |> Array.iteri (fun ind feature -> 
                            sprintf "|%s %.3f |%s %.3f" labelsStreamName (labelsData.[ind]) featureStreamName feature
                            |> fun line -> builder.AppendLine(line)
                            |> ignore
                   )
        let trainDataFile = Path.Combine(__SOURCE_DIRECTORY__, "FunctionApproximation.txt")
        File.WriteAllText(trainDataFile, builder.ToString())
        let learningSource: DataSource = {
            SourcePath = trainDataFile
            Streams = [
                      featureStreamName, inputDim
                      labelsStreamName, numOutputClasses
                ]
            }

        let minibatchSource = textSource learningSource InfinitelyRepeat
        let minibatchHandler = fileMinibatchHandler minibatchSource (featureStreamName, labelsStreamName) config spec
    
        learner.learn config spec minibatchHandler
sw.Stop()
printfn "Time elapsed: %s" (sw.Elapsed.ToString())

#load "CNTK.fsx"
let xData = [0..minibatchSize-1] |> Seq.map (fun entry -> (fnc_x entry))
let yData = predictor |> CNTK.Debug.valueAt xData |> Seq.concat
