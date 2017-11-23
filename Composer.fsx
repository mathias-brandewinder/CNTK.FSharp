#load "CNTK.fsx"
open CNTK

open System
open System.Collections.Generic

// Variable kind: constant, input, output, parameter, placeholder

type PArguments =
    {
        Shape: int seq
        DataType:DataType
        InitialValue:float
    }

type CreateParam = DeviceDescriptor -> Parameter
type CreateInput = DeviceDescriptor -> Variable

type Unary = Variable -> Function
type NamedUnary = (Variable * string) -> Function
type Binary = (Variable * Variable) -> Function
type NamedBinary = (Variable * Variable * string) -> Function

type FunctionCompose () =
    static member Compose (left:CreateParam, right:CreateParam, f:Binary) = 
        fun device -> f(left device, right device)
    static member Compose (left:CreateParam, right:CreateParam, f:NamedBinary, name:string) = 
        fun device -> f(left device, right device, name)
    static member Compose (v:CreateParam, f:Unary) = 
        fun device -> f(v device)
    static member Compose (v:CreateParam, f:NamedUnary, name:string) = 
        fun device -> f(v device,name)

let param : PArguments -> DeviceDescriptor -> Parameter =
    fun args ->
        fun device ->    
            new Parameter(shape (args.Shape), args.DataType, args.InitialValue, device)

let p1 : CreateParam = param { Shape = [1;2;3]; DataType = DataType.Float; InitialValue = 0.0 }
let v1 : CreateInput = 
    fun device -> Variable.InputVariable(shape [1;2;3], DataType.Float)

let foo =
    fun device ->
        CNTKLib.ElementTimes(p1 device, v1 device)

let trainOn (device:DeviceDescriptor) (f:DeviceDescriptor->Function) = f device

let trainable = foo |> trainOn DeviceDescriptor.CPUDevice

let test = FunctionCompose.Compose (p1,p1,CNTKLib.ElementTimes,"product")
let testOnCPU = test |> trainOn DeviceDescriptor.CPUDevice
testOnCPU.Name
