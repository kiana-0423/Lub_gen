find_package(Qt6 REQUIRED COMPONENTS Widgets Sql)

add_executable(lub_gen
    main.cpp
    mainwindow.cpp
    csvparser.cpp
)

target_link_libraries(lub_gen PRIVATE Qt6::Widgets Qt6::Sql)
