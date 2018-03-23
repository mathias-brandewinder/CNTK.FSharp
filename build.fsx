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

// Build output directory
let buildDir = "./build/"

let project = [ "./CNTK.FSharp/CNTK.FSharp.fsproj" ]

// nuget package directory
let nugetDir = "./nuget/"

module Nuget = 

    let authors = [ "Mathias Brandewinder" ]
    let project = projectName
    let title = "F# extensions for CNTK"
    let summary = "F# extensions to simplify CNTK usage"
    let description = "F# extensions to simplify CNTK usage"
    let version = "0.1.2"
    let tags = "f#, fsharp, cntk, machine-learning, deep-learning"
    

(*
Build target / steps
*)

Target "RestorePackages" RestorePackages

// Clean contents from the build directory
Target "Clean" (fun _ ->
    CleanDir buildDir
    )

Target "LocalBuild" (fun _ ->
    project
    |> MSBuild buildDir "Rebuild" ["Platform", "x64"]  
    |> Log "Build Output: "
    )

Target "ReleaseBuild" (fun _ ->
    // Copy the dependency loading script into buildDir
    Path.Combine(nugetDir,"Dependencies.fsx")
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
            Authors = Nuget.authors
            Project = Nuget.project
            Title = Nuget.title
            Summary = Nuget.summary
            Description = Nuget.description
            Version = Nuget.version
            Tags = Nuget.tags
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
                    "Dependencies.fsx", Some "scripts/", None
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
