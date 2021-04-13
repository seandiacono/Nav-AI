const Url = "http://127.0.0.1:8000/";

var arrived = false;

$(
  setInterval(function () {
    requestUrl = Url + "get_trip_details";

    if (!arrived) {
      $.ajax({
        type: "GET",
        url: requestUrl,
        crossDomain: true,
        dataType: "json",
        success: function (res) {
          progress = res["progress"];
          time = res["time_left"];
          status = res["status"];
          image = res["image"];

          if(image == "ready"){
              return
          }

          if(status == "Initialising"){
            $('#loadingModal').modal('show');
          }else if(status == "Landing"){
            $('#landingModal').modal('show');
          }else{
            $('#loadingModal').modal('hide');
            $('#landingModal').modal('hide');
          }

          var d = new Date();
          d.setSeconds(d.getSeconds() + time);

          hour = d.getHours();
          minutes = d.getMinutes();

          $(".progress-bar")
            .css("width", progress + "%")
            .attr("aria-valuenow", progress);
          $("#time-left").html(time + " SEC");
          $("#current-action").html("Current Action: " + status);
          $("#eta").html("ETA: " + hour + ":" + minutes);
          $("#stream").attr("src", "data:image/jpeg;base64, " + image);

          if (status == "Home") {
            window.location.href = "donePage.html";
          }
        },
        error: function (jqXHR, textStatus, errorThrown) {
          console.log(errorThrown);
        },
      });
    }
  }, 1000)
);
