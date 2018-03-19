(*
FAKE Build Script
*)

#r @"packages/FAKE/tools/FakeLib.dll"
open Fake
open System
open System.IO

(*
Build configuration
*)

Environment.CurrentDirectory <- __SOURCE_DIRECTORY__

let projectName = "CNTK.FSharp"
let projectDescription = "F# extensions to simplify CNTK usage"
let projectSummary = "F# extensions to simplify CNTK usage"

// Build output directory
let buildDir = "./build/"

let project = [ "./CNTK.FSharp/CNTK.FSharp.fsproj" ]

// nuget package directory
let nugetDir = "./nuget/"


(*
Build target / steps
*)

Target "RestorePackages" RestorePackages

// Clean contents from the build directory
Target "Clean" (fun _ ->
    CleanDir buildDir
    )

Target "LocalBuild" (fun _ ->
    // Copy the dependency loading script into buildDir
    "ScriptLoader.fsx"
    |> CopyFile buildDir
    // Build the project to buildDir
    project
    |> MSBuild buildDir "Rebuild" ["Platform", "x64"]  
    |> Log "Build Output: "
    )

Target "ReleaseBuild" (fun _ ->
    // Copy the dependency loading script into buildDir
    "ScriptLoader.fsx"
    |> CopyFile buildDir
    // Build the project to buildDir
    project
    |> MSBuildReleaseExt buildDir ["Platform", "x64"] "Rebuild" 
    |> Log "Build Output: "
    )

Target "CreateNuget" (fun _ ->
        
    "./nuget/" + projectName + ".nuspec"
    |> NuGet (fun p ->
        { p with
            Authors = [ "Mathias Brandewinder" ]
            Project = projectName
            Title = "F# extensions for CNTK"
            Summary = projectSummary
            Description = projectDescription
            Version = "0.1.1-alpha"
            Tags = "f#, fsharp, cntk, machine-learning, deep-learning"
            WorkingDir = buildDir
            OutputPath = nugetDir
            Dependencies = 
                [ 
                    "CNTK.CPUOnly", "2.5.0" 
                ]
            References = 
                [ 
                    "CNTK.FSharp.dll"
                ]
            Files = 
                [ 
                    "CNTK.FSharp.dll", Some "lib/", None
                    "ScriptLoader.fsx", Some "content/", None
                ]
        })
    )    

(*
Build step dependencies
*)

"RestorePackages"
    ==> "Clean"
    ==> "LocalBuild"

"RestorePackages"
    ==> "Clean"
    ==> "ReleaseBuild"
    ==> "CreateNuget"

RunTargetOrDefault "LocalBuild"
