$("#destination_form").submit(function(e) {
    e.preventDefault();
});

const Url = "http://127.0.0.1:8000/";

$('#destination_form').on("submit", function(e) {  

    var x = document.forms['destinationForm']['xCoord'].value;
    var y = document.forms['destinationForm']['yCoord'].value;
    var altitude = document.forms['destinationForm']['altitude'].value;
    var velocity = document.forms['destinationForm']['velocity'].value;

    requestUrl = Url + "sendFlightParams";

    console.log(JSON.stringify({
        "xCoord": x,
        "yCoord": y,
        "altitude": altitude,
        "velocity": velocity
    }));
    $.ajax
    ({
        type: "POST",
        url: requestUrl,
        crossDomain: true,
        contentType: "application/json",
        data: JSON.stringify({
            "xCoord": x,
            "yCoord": y,
            "altitude": altitude,
            "velocity": velocity
        }),
        success: function () {
            window.location.href = "videostream.html";
        },
        error: function(jqXHR, textStatus, errorThrown) { 
            console.log(errorThrown)
        }
    })

});