# Building OptSched with LLVM 6 and Clang

### [Table of Contents]

+ **[Setup]**

  + **[Ubuntu][setup-ubuntu]**

    + **[Update Packages][setup-ubuntu-update]**

    + **[Install CMake and Git][setup-ubuntu-cmake-git]**

    + **[Install Ninja (optional)][setup-ubuntu-ninja]**

  + **[MacOS][setup-macos]**

    + **[Install Command Line Developer Tools][setup-macos-dev-tools]**

    + **[Install Homebrew (optional)][setup-macos-homebrew]**

    + **[Install Cmake][setup-macos-cmake]**

    + **[Install Ninja (optional)][setup-macos-ninja]**

+ **[Super-build Script][super-build]**

+ **[Manual Build][manual-build]**

  + **[Download the Source Code][download-source]**

  + **[Build LLVM, Clang and OptSched][build]**

    + **[Command Line Build (Ubuntu and MacOS)][build-cli]**

    + **[MacOS Xcode Build][build-xcode]**

+ **[Test the Build][test]**

---

## Setup

###### Only Ubuntu and MacOS are known to work, but these instructions are likely to work other platforms. Please let us know if you successfully build on a different platform, so we can improve these instructions.

### Ubuntu

_**Attention:** Please only run these instructions on your own machine. If you are building on **Grace 2**, please skip_
_forward to [Download the Source Code][download-source]._

###### Starting with a fresh install of [Ubuntu 16.04] is recommended.

#### Update Packages

###### This may not be necessary, but is often a good idea.

`
sudo apt update && sudo apt upgrade
`

#### Install [CMake] and [Git]

`
sudo apt install cmake git
`

#### Install [Ninja] (optional)


It is recommended to build LLVM using Ninja to avoid running out of memory during linking. Using Ninja should also result in faster builds.


##### Downloading manually (recommended)

`
wget -q https://github.com/ninja-build/ninja/releases/download/v1.9.0/ninja-linux.zip && unzip -q ninja-linux.zip && sudo cp ninja /usr/bin && rm ninja ninja-linux.zip
`

##### Using APT (Ubuntu 18.04 or later)

###### Note: On Ubuntu 16.04 the version of Ninja installed by APT is too old and will not work.

`
apt install ninja
`

Proceed to [Download the Source Code][download-source].

### MacOS

#### Install Command Line Developer Tools

Open `Terminal` and run

`
xcode-select --install
`

and select either `Install` or `Get Xcode`, if you want to install `Xcode` and have not already.

#### Install [Homebrew] (optional)

Homebrew is a package manager for MacOS - it provides a simple way to install and manage software on your Mac.

To install homebrew, open `Terminal` and run

`
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
`

#### Install [CMake]

##### Using Homebrew (preferred)

`
brew install cmake
`

##### Using CMake Installer

Visit the [CMake downloads page], and download the file ending in `Darwin-x86_64.dmg`.

Open the downloaded image, and drag the `CMake` application into the `Applications` folder.

#### Install [Ninja] (optional)

It is recommended to build LLVM using Ninja to avoid running out of memory during linking. Using Ninja should also result in faster builds.

##### Using Homebrew (preferred)

`
brew install ninja
`

##### Downloading manually

Download and install Ninja 1.9:

`
wget -q https://github.com/ninja-build/ninja/releases/download/v1.9.0/ninja-mac.zip && unzip -q ninja-mac.zip && sudo cp ninja /usr/bin && rm ninja ninja-mac.zip
`

#### Install [Xcode] (optional)

If you would like to use Xcode to build LLVM, and do not already have it installed, go to the Mac App Store,
search for `Xcode`, and click `Get`.

## Super-build Script

To let a script manage cloning and installing all dependencies, placing OptSched inside llvm for you.

### Configure with CMake

**1. Create a build directory.**

`
mkdir build && cd build
`

**2. Configure**

We want to configure against the OptSched/cmake/superbuild directory.
Use the generator you want, be it Ninja with `-GNinja`, explicitly specified makefiles with `-G'Unix Makefilex'`,
or something else.

If you have ccache installed, consider adding `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`.
This will speed up subsequent builds. If you do so, be sure to disable the ccache `hash_dir` setting.

To build OptSched inside LLVM:

`
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -DLLVM_PARALLEL_LINK_JOBS=1 ../cmake/superbuild
`

###### Note: In debug builds, linking uses a lot of memory. Set `LLVM_PARALLEL_LINK_JOBS=2` if you have >= 32G memory, otherwise use `LLVM_PARALLEL_LINK_JOBS=1`.

If you also wish to build flang, add `-DOPTSCHEDSUPER_FLANG=ON`.
The flang compiler cannot be built with ninja, so if you are using Ninja, add `-DOPTSCHEDSUPER_FLANG_COMPILER_CMAKE_GENERATOR='Unix Makefiles'`

Complete command for building with flang:

`
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -DLLVM_PARALLEL_LINK_JOBS=1 -DOPTSCHEDSUPER_FLANG_COMPILER_CMAKE_GENERATOR='Unix Makefiles' -DOPTSCHEDSUPER_FLANG=ON ../cmake/superbuild
`

**3. Run the Super-build**

Invoke the generator you chose.

Ninja:

`
ninja
`

Make:

`
make # Consider adding -jN where N is the number of parallel compile processes you want
`

Generic:

`
cmake --build .
`

The CMake super-build script will clone, configure, and build llvm-project along with flang if specified.
The main build directory for LLVM, where unit tests can be run, is `llvm-prefix/src/llvm-build`.

The flang binaries and libraries will be installed to `flang-install` inside your build directory.
The llvm binaries and libraries, including OptSched.so, will be installed to `llvm-install` inside your build directory.
These directories may be changed at the configure step by specifying `-DOPTSCHEDSUPER_<Type>_INSTALL_PREFIX=/path/to/install/dir`,
where `<Type>` is either `LLVM` or `FLANG`.

## Manual Build

To manually build this, such as if you want to place OptSched inside an existing clone of llvm.

### Download the Source Code

**1. Clone the [LLVM source code] from GitHub.**

`
git clone https://github.com/llvm/llvm-project
`

**2. Checkout LLVM release 6.**

`
cd llvm-project && git checkout release/6.x
`

**3. Clone OptSched into the projects directory.**

`
cd llvm/projects && git clone https://github.com/CSUS-LLVM/OptSched
`

**4. Create a build directory.**

`
mkdir build && cd build
`

**5. Apply [this patch][spilling-info-patch] to print spilling info.**

`
git am ../OptSched/patches/llvm6.0/llvm6-print-spilling-info.patch
`

### Build LLVM, Clang and OptSched

#### Command Line Build (Ubuntu and MacOS)

###### These instructions follow after [Download the Source Code][download-source], and so assume that you are in the `llvm-project/llvm/projects/build` directory.

**Using Ninja (recommended)**

###### Note: In debug builds, linking uses a lot of memory. Set `LLVM_PARALLEL_LINK_JOBS=2` if you have >= 32G memory, otherwise use `LLVM_PARALLEL_LINK_JOBS=1`.

`
cmake -GNinja -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_ENABLE_PROJECTS='clang' -DCMAKE_BUILD_TYPE=Debug '-DLLVM_TARGETS_TO_BUILD=X86' -DLLVM_BUILD_TOOLS=ON -DLLVM_INCLUDE_TESTS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON ../..
`

`
ninja
`

**Using Make**

###### Note: Debug builds use a lot of memory. The build will fail if you do not have enough. If this happens, try using Ninja to build.

`
cmake -DLLVM_ENABLE_PROJECTS='clang' -DCMAKE_BUILD_TYPE=Debug '-DLLVM_TARGETS_TO_BUILD=X86' -DLLVM_BUILD_TOOLS=ON -DLLVM_INCLUDE_TESTS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON ../..
`

`
make
`

_A Debug build of LLVM on a single thread will take a long time._

_See [Building with CMake] for more build options._

#### MacOS Xcode Build

###### These instructions follow after [Download the Source Code][download-source], and so assume that you are in the `llvm-project/llvm/projects` directory.

**1. Build an Xcode project**

`
cmake -G Xcode -DLLVM_ENABLE_PROJECTS='clang' -DCMAKE_BUILD_TYPE=Debug '-DLLVM_TARGETS_TO_BUILD=X86' -DLLVM_BUILD_TOOLS=ON -DLLVM_INCLUDE_TESTS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON ../..
`

This will create an Xcode project in `llvm-project/llvm/projects/build`.

**2. Open the project in Xcode**

Open Xcode, and go to `File > Open...` (or press `Cmd+O`).

Navigate to `llvm-projects/llvm/projects/build` and click `Open`.

**3. Create `clang` and `OptSched` schemes**

Upon opening the project, Xcode will prompt you to create schemes. We want to create them manually - this will prevent you from having to sort through tons of build targets later.

Press the `+` button and select `clang` from the long `Targets` dropdown list.

###### Note: Typing `clang` will filter the dropdown list, making it easier to find.

Repeat this step, selecting `OptSched` from the list.

You should now have two schemes - `clang` and `OptSched`. Press `Close`.

**4. Change the language standard to C++14**

In the file navigation window on the left, scroll to the very top and select the `LLVM`
project file, under which all other files should be listed. It has a blue icon.

Just to the right of the file navigator there will be a list of all build targets.
Select `OptSched` from this list.

###### Note: Typing `OptSched` in the `Filter` box will help you locate it more easily.

Click on `Build Settings` at the top of the page, and scroll down to the
`Apple Clang - Custom Compiler Flags` section.

Under `Other C++ Flags`, double-click on the long list of compiler flags to the right of `Debug`.
From the text box that pops up, double-click on `-std=c++11` and change it to `-std=c++14`.

###### Note: If you want to compile in `Release` mode, or any of the others, you will have to make the same change for that mode.

**5. Build `clang` and `OptSched`**

At the top left, there is a "Run" (►) button and a "Stop" (■) button. To the right of those is where you can chose your scheme.

Select the `clang` scheme and then select `Product > Build` (or press `Cmd + B`).

Wait for the build to complete, and repeat this with the `OptSched` scheme.

## Test the Build

**Super-build**

Run ctest with `make test` (or `ninja test`) or by running ctest directly with `ctest .` .
This will run the unit tests and it will do a test run of the compiler with OptSched enabled.

**Unit Tests**

Invoke the `check-optsched-unit` target of the build system generated by CMake.
For `make`, that is `make check-optsched-unit`. In general, that is `cmake --build . --target check-optsched-unit`

If you used the superbuild, you need to be in the `llvm-prefix/src/llvm-build` directory first.

**Command Line Build**

From the `llvm-project/llvm/projects/build` directory, run:

`
echo 'int main(){};' | ./bin/clang -xc - -O3 -fplugin=lib/OptSched.so -mllvm -misched=optsched -mllvm -enable-misched -mllvm -optsched-cfg=../OptSched/example/optsched-cfg -mllvm -debug-only=optsched
`

**MacOS Xcode Build**

From the `llvm-project/llvm/projects/build` directory, run:

`
echo 'int main(){};' | Debug/bin/clang -xc - -O3 -fplugin=lib/OptSched.so -mllvm -misched=optsched -mllvm -enable-misched -mllvm -optsched-cfg=../OptSched/example/optsched-cfg -mllvm -debug-only=optsched
`

<!-- TODO: Show expected output of test command -->


[table of contents]: #table-of-contents
[setup]: #setup
[setup-ubuntu]: #ubuntu
[setup-ubuntu-update]: #update-packages
[setup-ubuntu-cmake-git]: #install-cmake-and-git
[setup-ubuntu-ninja]: #install-ninja-optional
[setup-macos]: #macos
[setup-macos-dev-tools]: #install-command-line-developer-tools
[setup-macos-homebrew]: #install-homebrew-optional
[setup-macos-cmake]: #install-cmake
[setup-macos-ninja]: #install-ninja-optional-1
[super-build]: #super-build-script
[manual-build]: #manual-build
[download-source]: #download-the-source-code
[build]: #build-llvm-clang-and-optsched
[build-cli]: #command-line-build-ubuntu-and-macos
[build-xcode]: #macos-xcode-build
[test]: #test-the-build

<!-- Outside links -->
[ubuntu 16.04]: http://releases.ubuntu.com/16.04/
[cmake]: https://cmake.org/
[git]: https://git-scm.com/
[homebrew]: https://brew.sh/
[cmake downloads page]: https://cmake.org/download/
[xcode]: https://developer.apple.com/xcode/
[ninja]: https://ninja-build.org/
[llvm source code]: https://github.com/llvm/llvm-project
[building with cmake]: https://llvm.org/docs/CMake.html
[spilling-info-patch]: patches/llvm6.0/llvm6-print-spilling-info.patch
