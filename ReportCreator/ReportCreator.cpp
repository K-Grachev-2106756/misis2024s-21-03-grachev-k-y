#include "ReportCreator.h"




void ReportCreator(const std::string& labName, const std::string& taskText) {
    std::string exportPath = "../export/" + labName;

    // Opening .cpp file
    std::ifstream codeFile("../prj.lab/" + labName + "/" + labName + ".cpp");
    if (!codeFile.is_open()){
        return;
    }

    // Opening file stream
    std::ofstream report(exportPath + "/report.md");
    if (!report.is_open()) {
        codeFile.close();
        return;
    }

    // Writing task
    report << 
    "# Report for " + labName << std::endl <<
    "## Task:\n" << taskText << std::endl;
    
    // Writing code
    report << "## Code:\n```";
    std::string line;
    while (std::getline(codeFile, line)) {
        report << line << std::endl;
    }
    report << "```" << std::endl;

    // Searching for images in export folder
    std::vector<std::filesystem::path> pngFiles;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(exportPath)) {
            if (entry.path().extension() == ".png") {
                pngFiles.push_back(entry.path().filename());
            }            
        } 
    } 
    catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << std::endl;
    }

    // Writing md-structure for image displaying
    report << "## Results:\n";
    for (const auto& name : pngFiles) { 
        report << "![" << name << "](" << name.u8string() << ")" << std::endl;
    }     
}