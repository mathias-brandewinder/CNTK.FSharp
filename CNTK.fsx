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

#r "./packages/CNTK.CPUOnly/lib/net45/x64/Cntk.Core.Managed-2.2.dll"
open CNTK

// utilities

type Shape = NDShape
let shape (dims:int seq) = NDShape.CreateNDShape dims

type GlorotParameters = {
        Scale:float
        OutputRank:int
        FilterRank:int
        Seed:int
    }
let defaultGlorotParams = {
    Scale = float CNTKLib.DefaultParamInitScale
    OutputRank = CNTKLib.SentinelValueForInferParamInitRank
    FilterRank = CNTKLib.SentinelValueForInferParamInitRank
    Seed = 1
    }

type Init = 
    | GlorotUniform of GlorotParameters

type Initialization = 
    | Value of float
    | Initializer of Init

type Model = 
    | Input of Variable
    | Param of Shape
    | Add of Model * Model
    | Prod of Model * Model
    | Named of string * Model
    static member (+) (left:Model,right:Model) = Add(left,right)
    static member (*) (left:Model,right:Model) = Prod(left,right)

let variable v = new Variable(v)

type VorF = 
    | V of Variable 
    | F of Function
    member this.ToVar =
        match this with
        | V(v) -> v
        | F(f) -> new Variable(f)
    member this.ToFun =
        match this with
        | V(v) -> v.ToFunction()
        | F(f) -> f

let buildFor (device:DeviceDescriptor) (model:Model) =

    let rec build (name:string option) (model:Model)  =
        match model with
        | Named(name,model) ->
            build (Some name) model
        | Input(inputVariable) -> V(inputVariable)
        | Param(s) ->            
            match name with
            | None -> V(new Parameter(s, DataType.Float, 0.0, device))
            | Some(name) -> V(new Parameter(s, DataType.Float, 0.0, device, name))
        | Add(left,right) ->
            let left = build None left
            let right = build None right
            match name with
            | None -> F(CNTKLib.Plus(left.ToVar,right.ToVar))
            | Some(name) -> F(CNTKLib.Plus(left.ToVar,right.ToVar,name))
        | Prod(left,right) ->
            let left = build None left
            let right = build None right
            match name with
            | None -> F(CNTKLib.Times(left.ToVar,right.ToVar))
            | Some(name) -> F(CNTKLib.Times(left.ToVar,right.ToVar,name))
    
    build None model

let named name model = Named(name,model)

let rec dim (model:Model) =
    match model with
    | Input(v) -> v.Shape
    | Param(s) -> s
    | Add(left,right) -> 
        // could add check, dims should match
        dim left
    | Prod(left,right) ->
        let first = (dim left).Dimensions |> Seq.head
        let second = (dim right).Dimensions |> Seq.last
        shape [ first; second ]
    | Named(_,model) -> dim model
    
type Evaluation = 
    | CrossEntropyWithSoftmax
    | ClassificationError

type Predictor = Variable -> Model

let trainerOn (device:DeviceDescriptor) (features:Variable,labels:Variable,model:Predictor,loss:Evaluation,eval:Evaluation) =
    
    let classifier = 
        model features 
        |> buildFor device
        |> fun x -> x.ToFun

    let loss = 
        match loss with
        | CrossEntropyWithSoftmax -> CNTKLib.CrossEntropyWithSoftmax(variable classifier, labels)
        | ClassificationError ->  CNTKLib.ClassificationError(variable classifier, labels)

    let eval = 
        match eval with
        | CrossEntropyWithSoftmax -> CNTKLib.CrossEntropyWithSoftmax(variable classifier, labels)
        | ClassificationError ->  CNTKLib.ClassificationError(variable classifier, labels)

    // this should be an argument
    let learningRatePerSample = new TrainingParameterScheduleDouble(0.02, uint32 1) 
    let parameterLearners =
        ResizeArray<Learner>(
            [ 
                Learner.SGDLearner(classifier.Parameters(), learningRatePerSample) 
            ])

    Trainer.CreateTrainer(classifier, loss, eval, parameterLearners)

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