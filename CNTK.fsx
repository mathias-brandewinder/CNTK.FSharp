(*
This file is intended to load dependencies in an F# script,
to train a model from the scripting environment.
CNTK, CPU only, is assumed to have been installed via Paket.
*)

open System
open System.IO
open System.Collections.Generic

Environment.SetEnvironmentVariable("Path",
    Environment.GetEnvironmentVariable("Path") + ";" + __SOURCE_DIRECTORY__)

let dependencies = [
        "./packages/CNTK.CPUOnly/lib/net45/x64/"
        "./packages/CNTK.CPUOnly/support/x64/Dependency/"
        "./packages/CNTK.CPUOnly/support/x64/Dependency/Release/"
        "./packages/CNTK.CPUOnly/support/x64/Release/"    
    ]

dependencies 
|> Seq.iter (fun dep -> 
    let path = Path.Combine(__SOURCE_DIRECTORY__,dep)
    Environment.SetEnvironmentVariable("Path",
        Environment.GetEnvironmentVariable("Path") + ";" + path)
    )    

#I "./packages/CNTK.CPUOnly/lib/net45/x64/"
#I "./packages/CNTK.CPUOnly/support/x64/Dependency/"
#I "./packages/CNTK.CPUOnly/support/x64/Dependency/Release/"
#I "./packages/CNTK.CPUOnly/support/x64/Release/"

#r "./packages/CNTK.CPUOnly/lib/net45/x64/Cntk.Core.Managed-2.3.dll"
open CNTK

// utilities

let shape (dims:int seq) = NDShape.CreateNDShape dims

type Layer = DeviceDescriptor -> Variable -> Function

type Loss = 
    | CrossEntropyWithSoftmax
    | ClassificationError

let evalutation (loss:Loss) (predicted:Function,actual:Variable) =
    match loss with
    | CrossEntropyWithSoftmax -> CNTKLib.CrossEntropyWithSoftmax(new Variable(predicted),actual)
    | ClassificationError -> CNTKLib.ClassificationError(new Variable(predicted),actual)

type Schedule = {
    Rate:float
    MinibatchSize:int
    }

type Specification = {
    Features: Variable
    Labels: Variable
    Model: Layer
    Loss: Loss
    Eval: Loss
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

let prepare (device:DeviceDescriptor) (spec:Specification) =
    
    let predictor = spec.Model device spec.Features
    let loss = evalutation spec.Loss (predictor,spec.Labels)
    let eval = evalutation spec.Eval (predictor,spec.Labels)   
    let parameterLearners = learning predictor spec.Schedule     
    let trainer = Trainer.CreateTrainer(predictor, loss, eval, parameterLearners)
    
    predictor, trainer

[<RequireQualifiedAccess>]
module Layers =

    let stack (next:Layer) (curr:Layer) : Layer =
        fun device ->
            fun variable ->
                let intermediate = new Variable(curr device variable)
                next device intermediate

    let scaled<'T> (scalar:'T) : Layer = 
        fun device ->
            fun input ->
                CNTKLib.ElementTimes(Constant.Scalar<'T>(scalar, device), input)

    // TODO naming
    let dense (outputDim:int) : Layer =
        fun device ->
            fun input ->

                let input : Variable =
                    if (input.Shape.Rank <> 1)
                    then
                        let newDim = input.Shape.Dimensions |> Seq.reduce (*)
                        new Variable(CNTKLib.Reshape(input, shape [ newDim ]))
                    else input

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
                CNTKLib.Plus(plusParam, timesFunction)

[<RequireQualifiedAccess>]
module Activation =
    let ReLU : Layer = 
        fun device ->
            fun input ->
                CNTKLib.ReLU(input)
    let sigmoid : Layer =
        fun device ->
            fun input ->
                CNTKLib.Sigmoid(input)

    let tanh : Layer =
        fun device ->
            fun input ->
                CNTKLib.Tanh(input)

[<RequireQualifiedAccess>]
module Convolution =

    type Conv2D = {
        KernelWidth : int 
        KernelHeight : int 
        InputChannels : int
        OutputFeatures : int
        }
    let conv2D (args:Conv2D) : Layer = 
        fun device ->
            fun input ->
                let convWScale = 0.26

                let convParams = 
                    new Parameter(
                        shape [ args.KernelWidth; args.KernelHeight; args.InputChannels; args.OutputFeatures ], 
                        DataType.Float,
                        CNTKLib.GlorotUniformInitializer(convWScale, -1, 2), 
                        device)

                CNTKLib.Convolution(
                    convParams, 
                    input, 
                    shape [ 1; 1; args.InputChannels ]
                    )

    type Pool2D = {
        WindowWidth : int
        WindowHeight : int
        HorizontalStride : int 
        VerticalStride : int
        PoolingType : PoolingType
        }                
    let pooling2D (args:Pool2D) : Layer = 
        fun device ->
            fun input ->
                CNTKLib.Pooling(
                    input, 
                    args.PoolingType,
                    shape [ args.WindowWidth; args.WindowHeight ], 
                    shape [ args.HorizontalStride; args.VerticalStride ], 
                    [| true |])
let private dictAdd<'K,'V> (key,value) (dict:Dictionary<'K,'V>) = 
    dict.Add(key,value)
    dict

let dataMap xs = 
    let dict = Dictionary<Variable,Value>()
    xs |> Seq.fold (fun dict (var,value) -> dictAdd (var,value) dict) dict

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

let progress (trainer:Trainer, frequency) current =
    if current % frequency = 0
    then Some(current, minibatchSummary trainer)
    else Option.None

let printer = 
    Option.iter (fun (batch,report) ->
        printf "Batch: %i " batch
        printf "Average Loss: %.3f " (report.Loss)
        printf "Average Eval: %.3f " (report.Evaluation)
        printfn ""
        )

module Debug = 

    let debugDevice = DeviceDescriptor.CPUDevice

    let valueAt<'T> (value:'T seq) (f:Function) =

        let input = f.Inputs |> Seq.find (fun arg -> arg.IsInput)
        let inputValue = Value.CreateBatch(input.Shape, value, debugDevice)
        let output = f.Output
        let inputMap = 
            let map = Dictionary<Variable,Value>()
            map.Add(input,inputValue)
            map
        let outputMap =
            let map = Dictionary<Variable,Value>()
            map.Add(output, null)
            map
        f.Evaluate(inputMap,outputMap,debugDevice)
        outputMap.[output].GetDenseData<'T>(output) 
        |> Seq.map (fun x -> x |> Seq.toArray)
        |> Seq.toArray

    // TODO: fix this, not quite working
    let valueOf (name:string) (f:Function) =
        let param = 
            f.Inputs 
            |> Seq.tryFind (fun arg -> arg.Name = name)

        param
        |> Option.map (fun p ->
            let inputMap = Dictionary<Variable,Value>()
            let outputMap =
                let map = Dictionary<Variable,Value>()
                map.Add(p, null)
                map
            f.Evaluate(inputMap,outputMap,debugDevice)
            outputMap.[p].GetDenseData<float>(p) 
            |> Seq.map (fun x -> x |> Seq.toArray)
            |> Seq.toArray
            )
