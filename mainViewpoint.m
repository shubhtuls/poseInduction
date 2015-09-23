function errors = mainViewpoint(classInds, usePascalViews)
    %% Define Classes
    classes = {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','plant','sheep','sofa','train','tvmonitor'};
    %classInds = [1 2 4 5 6 7 9 11 14 18 19 20];
    %classInds = [1 2 4 5 6 7 9];
    %classInds = [1 2 6 7 9 14];
    %classInds = [1 2 4 5 6 7 9 11 14 18 19 20];
    %classInds = [1 2 4 5 6 7 9 14 18 19 20];
    %classInds = [12 17];
    
    numClasses = size(classInds,2);

    %% Iterate over pose predictions
    errors = zeros(numClasses,1);
    medErrors = zeros(numClasses,1);
    for c = 1:numClasses
        class = classes{classInds(c)};
        disp(class);
        [err,medErr] = regressToPose(class, usePascalViews(c));
        errors(c,:)=err;
        medErrors(c,:) = medErr;
    end
    prettyPrintResults(errors,medErrors);
end