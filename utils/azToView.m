function view = azToView(az)
az = az*180/pi;
if(az <= 45 || az >= 315)
    view = 'Frontal';
    return;
end

if(az >= 45 && az <= 135)
    view = 'Left';
    return;
end

if(az >= 135 && az <= 225)
    view = 'Rear';
    return;
end

view = 'Right';

end