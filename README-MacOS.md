# Building on MacOS with Xcode

You will need Xcode, which you can find in the Mac App Store.

1. Follow [these directions][quck-start] through step 4.

2. Build an Xcode project

```
cmake -G Xcode -DLLVM_ENABLE_PROJECTS='clang' -DCMAKE_BUILD_TYPE=Debug '-DLLVM_TARGETS_TO_BUILD=X86' -DLLVM_BUILD_TOOLS=ON -DLLVM_INCLUDE_TESTS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON ../..
```

This will create an Xcode project in llvm-project/llvm/projects/build.

3. Open the project in Xcode

Open Xcode, and go to `File > Open...` (or press `Cmd+O`).

Navigate to `llvm-projects/llvm/projects/build` and click `Open`.

4. Create `clang` and `OptSched` schemes

Upon opening the project, Xcode will prompt you to create schemes. We want to create them manually - this will prevent you from having to sort through tons of build targets later.

Press the `+` button and select `clang` from the long `Targets` dropdown list.

_Note: Typing `clang` will filter the dropdown list, making it easier to find_

Repeat this step, selecting `OptSched` from the list.

You should now have two schemes - `clang` and `OptSched`. Press `Close`.

5. Change the language standard to C++14

In the file navigation window on the left, scroll to the very top and select the `LLVM`
project file, under which all other files should be listed. It has a blue icon.

Just to the right of the file navigator there will be a list of all build targets.
Select `OptSched` from this list.

_Note: Typing `OptSched` in the `Filter` box will help you locate it more easily._

Click on `Build Settings` at the top of the page, and scroll down to the
`Apple Clang - Custom Compiler Flags` section.

Under `Other C++ Flags`, double-click on the long list of compiler flags to the right of `Debug`.
From the text box that pops up, double-click on `-std=c++11` and change it to `-std=c++14`.

_Note: If you want to compile in `Release` mode, or any of the others, you will have to make that change for that mode._

6. Build `clang` and `OptSched`

At the top left, there is a "play" button and a "stop" button. To the right of those is where you can chose your scheme.

Select the `clang` scheme and then select `Product > Build` (or press `Cmd + B`).

Wait for the build to complete, and repeat this with the `OptSched` scheme.

### Test the build

Return to [step 6](README.md#Test-the-build-ubuntu-and-macos) of the quick start guide to test the build.

[quick-start]: README.md#build-optsched-with-llvm-6-and-clang-ubuntu-and-macos
