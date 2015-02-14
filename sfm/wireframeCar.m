function wireframe = wireframeCar()
edges = {
    %third parameter is edge color parameter
    % g for L-R
    % b for R-R
    % y for L-L

    'R_HeadLight','L_HeadLight','g';
    'R_TailLight','L_TailLight','g';

    'R_HeadLight','R_F_WheelCenter','b';
    'L_HeadLight','L_F_WheelCenter','r';

    'R_TailLight','R_B_WheelCenter','b';
    'L_TailLight','L_B_WheelCenter','r';

    'R_B_WheelCenter','L_B_WheelCenter','g';
    'R_F_WheelCenter','L_F_WheelCenter','g';

    'R_B_WheelCenter','R_F_WheelCenter','b';
    'L_B_WheelCenter','L_F_WheelCenter','r';

    'L_F_RoofTop','L_B_RoofTop','r';
    'L_F_RoofTop','R_F_RoofTop','g';
    'R_B_RoofTop','L_B_RoofTop','g';
    'R_B_RoofTop','R_F_RoofTop','b';

    'L_F_RoofTop','L_SideviewMirror','r';
    'R_F_RoofTop','R_SideviewMirror','b';

    'R_SideviewMirror','L_SideviewMirror','g';

    'L_B_RoofTop','L_TailLight','r';
    'R_B_RoofTop','R_TailLight','b';

    'L_HeadLight','L_SideviewMirror','r';
    'R_HeadLight','R_SideviewMirror','b';
 };

faces = {
    'R_F_RoofTop', 'L_F_RoofTop', 'L_B_RoofTop';
    'L_B_RoofTop', 'R_B_RoofTop', 'R_F_RoofTop';

    'R_SideviewMirror', 'R_HeadLight', 'L_HeadLight';
    'L_HeadLight', 'L_SideviewMirror', 'R_SideviewMirror';

    'R_F_RoofTop', 'R_SideviewMirror', 'L_SideviewMirror';
    'L_SideviewMirror', 'L_F_RoofTop', 'R_F_RoofTop';

    'R_B_RoofTop', 'L_B_RoofTop', 'L_TailLight';
    'L_TailLight', 'R_TailLight', 'R_B_RoofTop';

    'R_HeadLight', 'R_F_WheelCenter', 'L_F_WheelCenter';
    'L_F_WheelCenter', 'L_HeadLight', 'R_HeadLight';

    'R_TailLight', 'L_TailLight', 'L_B_WheelCenter';
    'L_B_WheelCenter', 'R_B_WheelCenter', 'R_TailLight';

    'R_F_WheelCenter', 'R_B_WheelCenter', 'L_B_WheelCenter';
    'L_B_WheelCenter', 'L_F_WheelCenter', 'R_F_WheelCenter';

    'L_HeadLight', 'L_F_WheelCenter', 'L_SideviewMirror';
    'L_B_WheelCenter', 'L_TailLight', 'L_B_RoofTop';
    'L_SideviewMirror', 'L_F_WheelCenter', 'L_B_WheelCenter';
    'L_SideviewMirror', 'L_B_WheelCenter', 'L_B_RoofTop';
    'L_F_RoofTop', 'L_SideviewMirror', 'L_B_RoofTop';

    'R_HeadLight', 'R_SideviewMirror', 'R_F_WheelCenter';
    'R_B_WheelCenter', 'R_B_RoofTop', 'R_TailLight';
    'R_SideviewMirror' 'R_B_WheelCenter', 'R_F_WheelCenter',;
    'R_SideviewMirror', 'R_B_RoofTop', 'R_B_WheelCenter';
    'R_F_RoofTop', 'R_B_RoofTop', 'R_SideviewMirror';
    };

ground_parts = {'R_B_WheelCenter','R_F_WheelCenter','L_B_WheelCenter'};
wireframe.edges = edges;
wireframe.faces = faces;
wireframe.ground_parts = ground_parts;
end
