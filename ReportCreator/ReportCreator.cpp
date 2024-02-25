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
    std::vector<std::string> pngFiles;
    std::string searchPath = exportPath + "\\*.png";
    WIN32_FIND_DATA findData;
    HANDLE findHandle = FindFirstFile(searchPath.c_str(), &findData);
    if (findHandle != INVALID_HANDLE_VALUE) {
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                pngFiles.push_back(findData.cFileName);
            }
        } while (FindNextFile(findHandle, &findData) != 0);
        FindClose(findHandle);
    }

    // Writing md-structure for image displaying
    report << "## Results:\n";
    for (const auto& name : pngFiles) { 
        report << "![" << name << "](" << name << ")" << std::endl;
    }
}