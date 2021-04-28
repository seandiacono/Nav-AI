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

          if(status == "Initialising"){
            $('#landingModal').modal('hide');
            $('#loadingModal').modal('show');
          }else if(status == "Landing"){
            $('#loadingModal').modal('hide');
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
          $("#current-action").html(status);
          $("#eta").html("ETA: " + hour + ":" + minutes);
          if(image == ""){
            $("#stream").attr("src", "assets/sample_img.png");
          }else{
            $("#stream").attr("src", "data:image/jpeg;base64, " + image);
          }

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
