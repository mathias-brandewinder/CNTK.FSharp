type Shape = seq<int>

type Model = 
    | Input of Shape
    | Param of Shape
    | Add of Model * Model
    | Prod of Model * Model
    | Named of string * Model
    static member (+) (left:Model,right:Model) = Add(left,right)
    static member (*) (left:Model,right:Model) = Prod(left,right)
    
let named name model = Named(name,model)

let model = (Input [28;28] * Param [28;28]) + Param [28] |> named "MODEL"

#load "CNTK.fsx"
open CNTK

let device = DeviceDescriptor.CPUDevice

let variable v = new Variable(v)

let rec build (name:string option) (model:Model)  =
    match model with
    | Named(name,model) ->
        build (Some name) model
    | Input(s) ->
        match name with
        | None -> Variable.InputVariable(shape s, DataType.Float)
        | Some(name) -> Variable.InputVariable(shape s, DataType.Float, name)
    | Param(s) ->
        match name with
        | None -> (new Parameter(shape s, DataType.Float, 0.0, device)) :> Variable
        | Some(name) -> (new Parameter(shape s, DataType.Float, 0.0, device, name)) :> Variable
    | Add(left,right) ->
        let left = build None left
        let right = build None right
        match name with
        | None -> CNTKLib.Plus(left,right)
        | Some(name) -> CNTKLib.Plus(left,right,name)
        |> variable
    | Prod(left,right) ->
        let left = build None left
        let right = build None right
        match name with
        | None -> CNTKLib.Times(left,right)
        | Some(name) -> CNTKLib.Times(left,right,name)
        |> variable

let foo = build None model
foo.DataType
foo.ToFunction()