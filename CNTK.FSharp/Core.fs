namespace CNTK.FSharp

[<AutoOpen>]
module Core = 

    open System
    open System.Collections.Generic
    open CNTK

    let var f = new Variable(f)
    let shape (dims:int seq) = NDShape.CreateNDShape dims

    // Wrapper type to simplify the conversions between 
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

        static member flatten (t:Tensor) =
            let tensorShape = t.toVar.Shape
            if (tensorShape.Rank <> 1)
            then
                let newDim = tensorShape.Dimensions |> Seq.reduce (*)
                CNTKLib.Reshape(t.toVar, shape [ newDim ]) |> Fun
            else t

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

    type Initializer = 
        | Value of float
        | GlorotUniform
        | Custom of CNTK.CNTKDictionary

    type Param () =
        static member init (dims: int seq, dataType: DataType, init: Initializer) =
            fun (device:DeviceDescriptor) ->
                match init with
                | Value(x) -> new Parameter(shape dims, dataType, x)
                | GlorotUniform -> new Parameter(shape dims, dataType, CNTKLib.GlorotUniformInitializer())
                | Custom(args) -> new Parameter(shape dims, dataType, args)

    type Loss = 
        | CrossEntropyWithSoftmax
        | ClassificationError
        | SquaredError

    let evaluation (loss:Loss) (predicted:Function, actual:Variable) =
        match loss with
        | CrossEntropyWithSoftmax -> 
            CNTKLib.CrossEntropyWithSoftmax(var predicted,actual)
        | ClassificationError -> 
            CNTKLib.ClassificationError(var predicted,actual)
        | SquaredError -> 
            CNTKLib.SquaredError(var predicted,actual)

    type Optimizer =
        | SGD
        | MomentumSGD of momentum:float
        | AdaDelta of rho:float * epsilon:float
        | AdaGrad of needsMultiplier:bool
        | Adam of momentum:float * unitGain:bool * varianceMomentum:float * epsilon:float * adaMax:bool
        | RMSProp of gamma:float * inc:float * dec:float * max:float * min:float * needsMultiplier:bool
    
    let paramsVector (parameters:Parameter seq) = 
        let parametersVector = new ParameterVector()
        parameters |> Seq.iter (parametersVector.Add)
        parametersVector

    let learnWith 
        (optimizer:Optimizer, schedule:TrainingParameterScheduleDouble) 
            (parameters:Parameter seq) =
                // TODO: add AdditionalLearningOptions 
                match optimizer with
                | SGD -> 
                    Learner.SGDLearner(ResizeArray(parameters), schedule)
                | MomentumSGD(momentumTimeConstant) ->
                    Learner.MomentumSGDLearner(
                        ResizeArray(parameters), 
                        schedule, 
                        CNTKLib.MomentumAsTimeConstantSchedule(momentumTimeConstant), 
                        true)
                | AdaDelta(rho,epsilon) -> 
                    CNTKLib.AdaDeltaLearner(paramsVector parameters, schedule,rho,epsilon)                    
                | AdaGrad(needsMultiplier) -> 
                    CNTKLib.AdaGradLearner(paramsVector parameters, schedule, needsMultiplier)
                | Adam(momentum, unitGain, varianceMomentum, epsilon, adaMax) -> 
                    let momentum = new TrainingParameterScheduleDouble(momentum)
                    let varianceMomentum = new TrainingParameterScheduleDouble(varianceMomentum)
                    CNTKLib.AdamLearner(paramsVector parameters, schedule, momentum, unitGain, varianceMomentum, epsilon, adaMax)
                | RMSProp(gamma, inc, dec, max, min, needsMultiplier) -> 
                    CNTKLib.RMSPropLearner(paramsVector parameters, schedule, gamma, inc, dec, max, min, needsMultiplier)

    [<RequireQualifiedAccess>]
    module Minibatch =

        let isSweepEnd (minibatch: UnorderedMapStreamInformationMinibatchData) =
            minibatch.Values
            |> Seq.exists(fun a -> a.sweepEnd)

        let getValues (minibatchSource:MinibatchSource) (name: string) (minibatch: UnorderedMapStreamInformationMinibatchData) =
            minibatch.[minibatchSource.StreamInfo(name)].data

        let getDense (variable:Variable) (data:Value) =
            match variable.DataType with
            | DataType.Double -> 
                data.GetDenseData<float>(variable) 
                |> Seq.map (Seq.map float >> Array.ofSeq)
                |> Array.ofSeq
            | DataType.Float -> 
                data.GetDenseData<single>(variable) 
                |> Seq.map (Seq.map float >> Array.ofSeq)
                |> Array.ofSeq
            | DataType.Float16 -> failwith "unsupported data type"
            | DataType.UChar -> failwith "unsupported data type"
            | DataType.Unknown -> failwith "unsupported data type"        

        type TrainingSummary = {
            Loss:float
            Evaluation:float
            Samples:uint32
            TotalSamples:uint32
            }

        let summary (trainer:Trainer) =
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

        let basicPrint (summary:TrainingSummary) =
            printfn "Total: %-8i Batch: %3i Loss: %.3f Eval: %.3f"
                summary.TotalSamples
                summary.Samples
                summary.Loss
                summary.Evaluation

        

    [<RequireQualifiedAccess>]
    module Dict =

        let add<'K,'V> (key,value) (dict:Dictionary<'K,'V>) = 
            dict.Add(key,value)
            dict

    type DataMap = Dictionary<Variable,Value>

    let dataMap xs = 
        let dataMap = DataMap ()
        xs 
        |> Seq.fold (fun dict (var,value) -> 
            Dict.add (var,value) dict) dataMap        
