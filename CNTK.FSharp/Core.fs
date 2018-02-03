namespace CNTK.FSharp

[<AutoOpen>]
module Core = 

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

    let private dictAdd<'K,'V> (key,value) (dict:Dictionary<'K,'V>) = 
        dict.Add(key,value)
        dict

    let dataMap xs = 
        let dict = Dictionary<Variable,Value>()
        xs |> Seq.fold (fun dict (var,value) -> dictAdd (var,value) dict) dict
