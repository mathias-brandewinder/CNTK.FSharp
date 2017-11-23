#load "CNTK.fsx"
open CNTK

type CntkComputation<'R> = CNTK of (DeviceDescriptor -> 'R)

type CntkBuilder() = 
  member x.Return(v) = CNTK (fun _ -> v)
  member x.Bind(CNTK a, f) = 
      CNTK (fun d -> 
        let aval = a d
        let (CNTK bf) = f aval
        bf d)
  
let cntk = CntkBuilder()

type PArguments =
    {
        Shape: int seq
        DataType:DataType
        InitialValue:float
    }

let param (args : PArguments) =   
    CNTK(
        fun (device : DeviceDescriptor) -> 
            new Parameter(shape (args.Shape), args.DataType, args.InitialValue, device)
        )

let computation = 
    cntk {
        let! p1 = param { Shape = [1;2;3]; DataType = DataType.Float; InitialValue = 0.0 }
        let! p2 = param { Shape = [1;2;3]; DataType = DataType.Float; InitialValue = 0.0 }
        return CNTKLib.ElementTimes(p1,p2) }

let runOn (d:DeviceDescriptor) (f:CntkComputation<Function>) = 
    f |> function | CNTK(v) -> v d

let cpuComputation = computation |> runOn DeviceDescriptor.CPUDevice
