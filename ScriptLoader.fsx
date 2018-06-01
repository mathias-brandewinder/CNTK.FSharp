(*
This file is intended to load dependencies in an F# script,
to train a model from the scripting environment.
CNTK, CPU only, is assumed to have been installed via Paket.
*)

open System
open System.IO

Environment.SetEnvironmentVariable("Path",
    Environment.GetEnvironmentVariable("Path") + ";" + __SOURCE_DIRECTORY__)

let dependencies = [
        "./packages/CNTK.GPU/lib/net45/x64/"
        "./packages/CNTK.GPU/support/x64/Release/"
        "./packages/CNTK.Deps.MKL/support/x64/Dependency/"
        "./packages/CNTK.Deps.Cuda/support/x64/Dependency/"
        "./packages/CNTK.Deps.cuDNN/support/x64/Dependency/"
        "./packages/CNTK.Deps.OpenCV.Zip/support/x64/Dependency/"
        "./packages/CNTK.Deps.OpenCV.Zip/support/x64/Dependency/Release/"
    ]

dependencies 
|> Seq.iter (fun dep -> 
    let path = Path.Combine(__SOURCE_DIRECTORY__,dep)
    Environment.SetEnvironmentVariable("Path",
        Environment.GetEnvironmentVariable("Path") + ";" + path)
    )    

#I "./packages/CNTK.GPU/lib/net45/x64/"
#I "./packages/CNTK.GPU/support/x64/Release/"

#r "./packages/CNTK.GPU/lib/net45/x64/Cntk.Core.Managed-2.5.1.dll"
