(*
FAKE Build Script
*)

#r @"packages/FAKE/tools/FakeLib.dll"
open Fake
open System
open System.IO

let buildDir = "./build/"

let project = [ "./CNTK.FSharp/CNTK.FSharp.fsproj" ]

Target "RestorePackages" RestorePackages

Target "Clean" (fun _ ->
    CleanDir buildDir
    )

Target "Build" (fun _ ->
    MSBuildReleaseExt buildDir ["Platform", "x64"] "Rebuild" project
    |> Log "Build Output: "
    )

"RestorePackages"
    ==> "Clean"
    ==> "Build"

RunTargetOrDefault "Build"