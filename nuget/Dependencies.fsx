(*
This script file is intended to load CNTK dependencies 
in an F# script, to work with a CNTK model from a 
scripting environment such as VS Code.
CNTK.FSharp is assumed to have been installed via Paket,
so all dependencies are located in ./packages/.
You can use it from your script by including the following:

// adjust the path below if necessary
#I "./packages/CNTK.FSharp/"
#load "scripts/ScriptLoader.fsx"
#r "lib/CNTK.FSharp.dll"
open CNTK
open CNTK.FSharp

*)

open System
open System.IO

let dependencies = [
        "../../CNTK.CPUOnly/lib/net45/x64/"
        "../../CNTK.CPUOnly/support/x64/Release/"
        "../../CNTK.Deps.MKL/support/x64/Dependency/"
        "../../CNTK.Deps.OpenCV.Zip/support/x64/Dependency/"
        "../../CNTK.Deps.OpenCV.Zip/support/x64/Dependency/Release"
    ]

dependencies 
|> Seq.iter (fun dep -> 
    let path = Path.Combine(__SOURCE_DIRECTORY__,dep)
    Environment.SetEnvironmentVariable("Path",
        Environment.GetEnvironmentVariable("Path") + ";" + path)
    )

#I "../../CNTK.CPUOnly/lib/net45/x64/"
#I "../../CNTK.CPUOnly/support/x64/Release/"

#r "../../CNTK.CPUOnly/lib/net45/x64/Cntk.Core.Managed-2.5.1.dll"
