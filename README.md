# nbodylab

Properties:
- C/C++
  - Additional Include Directories: C:\raylib-5.5_win64_msvc16\include;%(AdditionalIncludeDirectories)
- Linker
  - General
    - Additional Library Directories: $(CudaToolkitLibDir);C:\raylib-5.5_win64_msvc16\lib;%(AdditionalLibraryDirectories)
    
  - Input
    - Additional Dependencies: cudart_static.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;raylib.lib;winmm.lib;%(AdditionalDependencies)
  - Command Line
    - Additional Options: /FORCE:MULTIPLE



