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

let shape (dims:int seq) = NDShape.CreateNDShape dims

type VarOrFun =
    | Var of Variable
    | Fun of Function
    member this.Variable =
        match this with
        | Var v -> v
        | Fun f -> new Variable(f)
    member this.Function =
        match this with
        | Var v -> v.ToFunction ()
        | Fun f -> f
    static member (+) (left:VarOrFun,right:VarOrFun) = 
        (left.Variable + right.Variable) |> Fun
    static member (*) (left:VarOrFun,right:VarOrFun) =
        CNTKLib.Times (left.Variable, right.Variable) |> Fun
    static member ( *.) (left:VarOrFun,right:VarOrFun) =
        CNTKLib.ElementTimes (left.Variable, right.Variable) |> Fun

type CntkComputation = | CNTK of (DeviceDescriptor -> VarOrFun)

let OnDevice (device:DeviceDescriptor) x = 
    x |> function | CNTK(f) -> f device

type CntkBuilder () = 
    member x.Return(v) = CNTK (fun _ -> v)
    member x.Bind(CNTK a, f) = 
        CNTK (fun d -> 
            let aval = a d
            let (CNTK bf) = f aval
            bf d)

let cntk = CntkBuilder()

type Param () =
    static member create (pshape:int seq, dataType:DataType, init:float, name:string) =
        CNTK(
            fun (device : DeviceDescriptor) -> 
                Var(new Parameter(shape pshape, dataType, init, device, name))
            )
    static member create (pshape:int seq, dataType:DataType, init:float) =
        CNTK(
            fun (device : DeviceDescriptor) -> 
                Var(new Parameter(shape pshape, dataType, init, device))
            )

module Layer = 
    
    let scale<'T> (scalar:'T) (input:VarOrFun) =
        CNTK(fun device ->
            Fun(
                CNTKLib.ElementTimes(
                    Constant.Scalar<'T>(scalar, device),
                    input.Variable
                    )
                )
            )

    let linear = 
        fun (outputDim:int) ->
            fun (input:VarOrFun) ->
                cntk {
                    let input =
                        if (input.Variable.Shape.Rank <> 1)
                        then
                            let newDim = input.Variable.Shape.Dimensions |> Seq.reduce (*)
                            new Variable(CNTKLib.Reshape(input.Variable, shape [ newDim ]))
                            |> Var
                        else input

                    let inputDim = input.Variable.Shape.[0]

                    let! weights = Param.create ([ outputDim; inputDim ], DataType.Float, 1.0, "w")
                    let! bias = Param.create ([ outputDim ], DataType.Float, 0.0, "b")
                    
                    return (weights * input) + bias                
                }

let crossEntropyWithSoftmax (predicted:VarOrFun,actual:VarOrFun) = 
    CNTKLib.CrossEntropyWithSoftmax(predicted.Variable, actual.Variable)
let classificationError (predicted:VarOrFun,actual:VarOrFun) = 
    CNTKLib.ClassificationError(predicted.Variable, actual.Variable)
      

[<RequireQualifiedAccess>]
module Conv2D = 

    type Kernel = {
        Width:int
        Height:int
        }

    type Conv2D = {
        Kernel:Kernel
        InputChannels:int
        OutputFeatureMap:int
        }

    type Window = {
        Width:int
        Height:int 
        }

    type Stride = {
        Horizontal:int
        Vertical:int
        }

    type Pooling2D = {
        PoolingType:PoolingType
        Window:Window
        Stride:Stride
        }

    let pooling
        (pooling:Pooling2D)
        (input:VarOrFun) = 

            CNTKLib.Pooling(
                input.Variable, 
                pooling.PoolingType,
                shape [ pooling.Window.Width; pooling.Window.Height ], 
                shape [ pooling.Stride.Horizontal; pooling.Stride.Vertical ], 
                [| true |])
            |> Fun

    let convolution 
        (device:DeviceDescriptor)
        (conv:Conv2D)
        (features:VarOrFun) =

            // parameter initialization hyper parameter
            let convWScale = 0.26

            let convParams = 
                new Parameter(
                    shape [ 
                        conv.Kernel.Width
                        conv.Kernel.Height
                        conv.InputChannels
                        conv.OutputFeatureMap
                        ], 
                    DataType.Float,
                    CNTKLib.GlorotUniformInitializer(convWScale, -1, 2), 
                    device)

            CNTKLib.Convolution(
                convParams, 
                features.Variable, 
                shape [ 1; 1; conv.InputChannels ])
            |> Fun


let private dictAdd<'K,'V> (key,value) (dict:Dictionary<'K,'V>) = 
    dict.Add(key,value)
    dict

let dataMap xs = 
    let dict = Dictionary<Variable,Value>()
    xs |> Seq.fold (fun dict (var,value) -> dictAdd (var,value) dict) dict

let MiniBatchDataIsSweepEnd(minibatchValues:seq<MinibatchData>) =
    minibatchValues 
    |> Seq.exists(fun a -> a.sweepEnd)

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