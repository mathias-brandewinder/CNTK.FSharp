(*
FAKE Build Script
*)
open System
open System.IO

#r @"packages/FAKE/tools/FakeLib.dll"
open Fake
open Fake.AssemblyInfoFile
open Fake.ReleaseNotesHelper

(*
Build configuration
*)

Environment.CurrentDirectory <- __SOURCE_DIRECTORY__

let projectName = "CNTK.FSharp"
let authors = [ "Mathias Brandewinder" ]
let title = "F# extensions for CNTK"
let summary = "F# extensions to simplify CNTK usage"
let description = "F# extensions to simplify CNTK usage"
let tags = "f#, fsharp, cntk, machine-learning, deep-learning"

let releaseNotes = 
    LoadReleaseNotes "RELEASE_NOTES.md" 

let version = releaseNotes.AssemblyVersion

let project = [ "./CNTK.FSharp/CNTK.FSharp.fsproj" ]
let assemblyInfo = [ "./CNTK.FSharp/AssemblyInfo.fs" ]

// Build output directory
let buildDir = "./build/"
// nuget package output directory
let nugetDir = "./nuget/"

    

(*
Build target / steps
*)

Target "AssemblyInfo" (fun _ ->
    for file in assemblyInfo do
        CreateFSharpAssemblyInfo file
            [ 
                Attribute.Title title
                Attribute.Product projectName
                Attribute.Description summary
                Attribute.Version version
                Attribute.FileVersion version
            ]
    )

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
            Authors = authors
            Project = projectName
            Title = title
            Summary = summary
            Description = description
            Version = version
            Tags = tags
            WorkingDir = buildDir
            OutputPath = nugetDir
            Dependencies = 
                [ 
                    "CNTK.GPU", "2.5.1" 
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
    ==> "AssemblyInfo"
    ==> "Clean"
    ==> "ReleaseBuild"
    ==> "CreateNuget"

RunTargetOrDefault "LocalBuild"
