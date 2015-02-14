function [wireframe] = wireframeAeroplane()
%WIREFRAMEAEROPLANE Summary of this function goes here
%   Detailed explanation goes here
edges = {
    'Top_Rudder','Bot_Rudder','g';
    'L_Stabilizer','Bot_Rudder','g';
    'L_Stabilizer','Top_Rudder','g';    
    'R_Stabilizer','Top_Rudder','g';
    'R_Stabilizer','Bot_Rudder','g';
    'Bot_Rudder','Bot_Rudder_Front','g';
    'Bot_Rudder_Front','NoseTip','b';
    'L_WingTip','Left_Wing_Base','r';
    'R_WingTip','Right_Wing_Base','r';
    'Right_Wing_Base','Left_Wing_Base','r';
};
wireframe.edges = edges;

end