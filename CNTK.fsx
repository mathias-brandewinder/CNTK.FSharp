#load "ScriptLoader.fsx"
open CNTK

open System
open System.Collections.Generic

// utilities

let var f = new Variable(f)
let shape (dims:int seq) = NDShape.CreateNDShape dims

// Wrapper types to simplify the conversions between 
// Variable and Function, and smoothly combine operations
type Tensor = 
    | Var of Variable
    | Fun of Function

    member this.toVar = 
        match this with
        | Var(v) -> v
        | Fun(f) -> new Variable(f)
    member this.toFun = 
        match this with
        | Var(v) -> v.ToFunction ()
        | Fun(f) -> f 

    static member toFunction (t:Tensor) = t.toFun
    static member toVariable (t:Tensor) = t.toVar

    static member from (v:Variable) = Var(v)
    static member from (f:Function) = Fun(f)

    static member (+) (t1:Tensor,t2:Tensor) = 
        CNTKLib.Plus(t1.toVar,t2.toVar)
        |> Fun
    static member (*) (t1:Tensor,t2:Tensor) = 
        CNTKLib.Times(t1.toVar,t2.toVar)
        |> Fun
    static member (.*) (t1:Tensor,t2:Tensor) = 
        CNTKLib.ElementTimes(t1.toVar,t2.toVar)
        |> Fun

    static member plus (t1:Tensor,t2:Tensor) =
        CNTKLib.Plus (t1.toVar,t2.toVar)
        |> Fun    
    static member plus (t1:Tensor,t2:Tensor,name:string) =
        CNTKLib.Plus (t1.toVar,t2.toVar,name)
        |> Fun
    static member times (t1:Tensor,t2:Tensor) =
        CNTKLib.Times (t1.toVar,t2.toVar)
        |> Fun        
    static member times (t1:Tensor,t2:Tensor,name:string) =
        CNTKLib.Times (t1.toVar,t2.toVar,name)
        |> Fun
    static member elementTimes (t1:Tensor,t2:Tensor) =
        CNTKLib.ElementTimes (t1.toVar,t2.toVar)
        |> Fun
    static member elementTimes (t1:Tensor,t2:Tensor,name:string) =
        CNTKLib.ElementTimes (t1.toVar,t2.toVar,name)
        |> Fun

    static member exp (x:Tensor) = CNTKLib.Exp (x.toVar) |> Fun 
    static member log (x:Tensor) = CNTKLib.Log (x.toVar) |> Fun
    static member sigmoid (x:Tensor) = CNTKLib.Sigmoid (x.toVar) |> Fun
    static member tanh (x:Tensor) = CNTKLib.Tanh (x.toVar) |> Fun
  
type Variable with
    member this.Tensor = Var(this)
    member this.FromTensor(t:Tensor) = Tensor.toVariable t

type Function with
    member this.Tensor = Fun(this)
    member this.FromTensor(t:Tensor) = Tensor.toFunction t

let private dictAdd<'K,'V> (key,value) (dict:Dictionary<'K,'V>) = 
    dict.Add(key,value)
    dict

let dataMap xs = 
    let dict = Dictionary<Variable,Value>()
    xs |> Seq.fold (fun dict (var,value) -> dictAdd (var,value) dict) dict

type Activation = 
    | None
    | ReLU
    | Sigmoid
    | Tanh

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
