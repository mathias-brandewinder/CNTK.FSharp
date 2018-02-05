#load "ScriptLoader.fsx"
open CNTK

open System
open System.Collections.Generic

// utilities

let shape (dims:int seq) = NDShape.CreateNDShape dims

let isSweepEnd (minibatchValues: seq<MinibatchData>) =
    minibatchValues 
    |> Seq.exists(fun a -> a.sweepEnd)

type Initializer = 
    | Value of float
    | GlorotUniform
    | Custom of CNTK.CNTKDictionary

type Param() =

    static member init (dims: int seq, dataType: DataType, init: Initializer) =
        fun (device:DeviceDescriptor) ->
            match init with
            | Value(x) -> new Parameter(shape dims, dataType, x)
            | GlorotUniform -> new Parameter(shape dims, dataType, CNTKLib.GlorotUniformInitializer())
            | Custom(f) -> new Parameter(shape dims, dataType, f)

type DataSource = {
    SourcePath: string
    Streams: (string * int) seq
    }

type StreamProcessing = 
    | FullDataSweep
    | InfinitelyRepeat
let textSource (data:DataSource) =
    let streams = 
        data.Streams
        |> Seq.map (fun (name, dim) -> 
            new StreamConfiguration(name, dim))
        |> ResizeArray
    fun (processing: StreamProcessing) ->
        MinibatchSource.TextFormatMinibatchSource(
            data.SourcePath, 
            streams, 
            match processing with
            | FullDataSweep -> MinibatchSource.FullDataSweep
            | InfinitelyRepeat -> MinibatchSource.InfinitelyRepeat)
            
// Sequential model

type Computation = DeviceDescriptor -> Variable -> Function

type Loss = 
    | CrossEntropyWithSoftmax
    | ClassificationError
    | SquaredError

let evaluation (loss:Loss) (predicted:Function, actual:Variable) =
    match loss with
    | CrossEntropyWithSoftmax -> 
        CNTKLib.CrossEntropyWithSoftmax(new Variable(predicted),actual)
    | ClassificationError -> 
        CNTKLib.ClassificationError(new Variable(predicted),actual)
    | SquaredError -> 
        CNTKLib.SquaredError(new Variable(predicted),actual)


type Specification = {
    Features: Variable
    Labels: Variable
    Model: Computation
    Loss: Loss
    Eval: Loss
    }

type Schedule = {
    Rate: float
    MinibatchSize: int
    }

type Config = {
    MinibatchSize: int
    Epochs: int
    Device: DeviceDescriptor
    Schedule: Schedule
    } 

let learning (predictor:Function) (schedule:Schedule) =   
    let learningRatePerSample = 
        new TrainingParameterScheduleDouble(schedule.Rate, uint32 schedule.MinibatchSize)
    let parameterLearners =
        ResizeArray<Learner>(
            [ 
                Learner.SGDLearner(predictor.Parameters(), learningRatePerSample) 
            ])
    parameterLearners

type TrainingMiniBatchSummary = {
    Loss:float
    Evaluation:float
    Samples:uint32
    TotalSamples:uint32
    }

let minibatchSummary (trainer:Trainer) =
    if trainer.PreviousMinibatchSampleCount () <> (uint32 0)
    then
        {
            Loss = trainer.PreviousMinibatchLossAverage ()
            Evaluation = trainer.PreviousMinibatchEvaluationAverage ()
            Samples = trainer.PreviousMinibatchSampleCount ()
            TotalSamples = trainer.TotalNumberOfSamplesSeen ()
        }
    else
        {
            Loss = Double.NaN
            Evaluation = Double.NaN
            Samples = trainer.PreviousMinibatchSampleCount ()
            TotalSamples = trainer.TotalNumberOfSamplesSeen ()
        }

let basicMinibatchSummary (summary:TrainingMiniBatchSummary) =
    printfn "Total: %-8i Batch: %3i Loss: %.3f Eval: %.3f"
        summary.TotalSamples
        summary.Samples
        summary.Loss
        summary.Evaluation
let inMemoryMinibatchHandler 
        (config:Config) 
        (spec:Specification) 
        getData
        =
        let device = config.Device
        let featureVariable = spec.Features
        let labelVariable = spec.Labels
        fun step ->
            let isSweepEnd, featuresData, labelsData = getData step
            let featureValue = Value.CreateBatch (featureVariable.Shape, featuresData, device)
            let labelValue = Value.CreateBatch (labelVariable.Shape, labelsData, device)
            let batch = 
                [ 
                    featureVariable, new MinibatchData(featureValue)
                    labelVariable, new MinibatchData(labelValue)
                ] 
                |> dict
            (isSweepEnd,batch)
let fileMinibatchHandler 
        (source:MinibatchSource) 
        (featureStreamName:string, labelsStreamName:string) 
        (config:Config) 
        (spec:Specification) =
        let device = config.Device

        let input = spec.Features
        let labels = spec.Labels
        let featureStreamInfo = source.StreamInfo(featureStreamName)
        let labelStreamInfo = source.StreamInfo(labelsStreamName)
        let minibatchSize = uint32 (config.MinibatchSize)
        fun _ -> 
                let minibatchData = source.GetNextMinibatch(minibatchSize, device)
                let batch = 
                    [
                        input, minibatchData.[featureStreamInfo]
                        labels, minibatchData.[labelStreamInfo]
                    ]
                    |> dict
                (isSweepEnd (minibatchData.Values),batch)
type Learner () =

    let progress = new Event<TrainingMiniBatchSummary> ()
    member this.MinibatchProgress = progress.Publish

    member this.learn 
        (config:Config) 
        (spec:Specification) 
        (nextBatch : int -> (bool*IDictionary<Variable, MinibatchData>)) =

        let device = config.Device

        let predictor = spec.Model device spec.Features
        let loss = evaluation spec.Loss (predictor,spec.Labels)
        let eval = evaluation spec.Eval (predictor,spec.Labels)   

        let parameterLearners = learning predictor config.Schedule     
        let trainer = Trainer.CreateTrainer(predictor, loss, eval, parameterLearners)
        
        let rec learnEpoch (step,epoch) = 

            minibatchSummary trainer
            |> progress.Trigger

            if epoch <= 0
            // we are done : return function
            then predictor
            else
                let isSweepEnd,arguments = nextBatch step
                let step = step + 1
                trainer.TrainMinibatch(arguments, device) |> ignore
                
                // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
                // Batching will not end. Each time minibatchSource completes an sweep (epoch),
                // the last minibatch data will be marked as end of a sweep. We use this flag
                // to count number of epochs.
                let epoch = 
                    if isSweepEnd
                    then epoch - 1
                    else epoch

                learnEpoch (step, epoch)

        learnEpoch (0, config.Epochs)

[<RequireQualifiedAccess>]
module Layer = 
    
    // Combine 2 Computation Layers into 1
    let stack (next:Computation) (curr:Computation) : Computation =
        fun device ->
            fun variable ->
                let intermediate = new Variable(curr device variable)
                next device intermediate

    // combine a sequence of Computation Layers into 1
    let sequence (computations: Computation seq) =
        computations
        |> Seq.reduce (fun acc c -> stack c acc)

    let scale<'T> (scalar:'T) : Computation = 
        fun device ->
            fun input ->
                CNTKLib.ElementTimes(Constant.Scalar<'T>(scalar, device), input)

    let dense (outputDim:int) : Computation =
        fun device ->
            fun input ->
               
                let input : Variable =
                    if (input.Shape.Rank <> 1)
                    then
                        let newDim = input.Shape.Dimensions |> Seq.reduce (*)
                        new Variable(CNTKLib.Reshape(input, shape [ newDim ]))
                    else input

                let inputDim = input.Shape.[0]
                let dataType = input.DataType

                let weights = 
                    new Parameter(
                        shape [outputDim; inputDim], 
                        dataType,
                        CNTKLib.GlorotUniformInitializer(
                            float CNTKLib.DefaultParamInitScale,
                            CNTKLib.SentinelValueForInferParamInitRank,
                            CNTKLib.SentinelValueForInferParamInitRank, 
                            uint32 1),
                        device, 
                        "weights")

                let product = 
                    new Variable(CNTKLib.Times(weights, input, "product"))

                let bias = new Parameter(shape [ outputDim ], 0.0f, device, "bias")
                CNTKLib.Plus(bias, product)

    let dropout (proba:float) : Computation = 
        fun device ->
            fun input ->
                CNTKLib.Dropout(input,proba)

[<RequireQualifiedAccess>]
module Activation = 

    let ReLU : Computation = 
        fun device ->
            fun input ->
                CNTKLib.ReLU(input)

    let sigmoid : Computation = 
        fun device ->
            fun input ->
                CNTKLib.Sigmoid(input)

    let tanh : Computation = 
        fun device ->
            fun input ->
                CNTKLib.Tanh(input)


[<RequireQualifiedAccess>]
module Conv2D = 
    
    type Kernel = {
        Width: int
        Height: int
        }

    type Conv2D = {
        Kernel: Kernel 
        OutputFeatures: int
        Initializer: Initializer
        }

    let conv2D = {
        Kernel = { Width = 1; Height = 1 } 
        OutputFeatures = 1
        Initializer = GlorotUniform
        }

    let convolution (args:Conv2D) : Computation = 
        fun device ->
            fun input ->
                let kernel = args.Kernel
                let inputChannels = input.Shape.Dimensions.[2]
                let convParams = 
                    device
                    |> Param.init (
                        [ kernel.Width; kernel.Height; inputChannels; args.OutputFeatures ], 
                        DataType.Float,
                        args.Initializer)

                CNTKLib.Convolution(
                    convParams, 
                    input, 
                    shape [ 1; 1; inputChannels ]
                    )

    type Window = {
        Width: int
        Height: int          
        }

    type Stride = {
        Horizontal: int
        Vertical: int
        }

    type Pool2D = {
        Window: Window
        Stride : Stride 
        PoolingType : PoolingType
        }                
    let pooling (args:Pool2D) : Computation = 
        fun device ->
            fun input ->

                let window = args.Window
                let stride = args.Stride

                CNTKLib.Pooling(
                    input, 
                    args.PoolingType,
                    shape [ window.Width; window.Height ], 
                    shape [ stride.Horizontal; stride.Vertical ], 
                    [| true |]
                    )
let private dictAdd<'K,'V> (key,value) (dict:Dictionary<'K,'V>) = 
    dict.Add(key,value)
    dict

let dataMap xs = 
    let dict = Dictionary<Variable,Value>()
    xs |> Seq.fold (fun dict (var,value) -> dictAdd (var,value) dict) dict
