function dependencyPath = check_mathworks_example_dependency()
%CHECK_MATHWORKS_EXAMPLE_DEPENDENCY Locate MathWorks example support file.
%   DEPENDENCYPATH = CHECK_MATHWORKS_EXAMPLE_DEPENDENCY() returns the path
%   to Example_DieboldLi.m when it is available on the MATLAB path.
%   Otherwise, the function raises an error with setup guidance.

dependencyName = "Example_DieboldLi";
dependencyPath = which(dependencyName);

if strlength(string(dependencyPath)) == 0
    error("check_mathworks_example_dependency:MissingFile", ...
        ['Missing MathWorks example support file %s.m. Open the ', ...
         'MathWorks Diebold-Li example once in MATLAB so that the ', ...
         'supporting files are downloaded locally, then confirm the path ', ...
         'using `which %s -all`.'], dependencyName, dependencyName);
end

fprintf("Using MathWorks example dependency: %s\n", dependencyPath);

end
