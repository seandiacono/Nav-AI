from flask import Flask, request, jsonify

app = Flask(__name__)


class FlightParams():

    def init_flight_params(self, x, y, altitude, velocity):
        self.x = x
        self.y = y
        self.altitude = altitude
        self.velocity = velocity


flight_params = FlightParams()


@app.route('/')
def index():
    return "Server Running"


@app.route('/sendFlightParams', methods=['POST'])
def set_flight_params():
    data = request.get_json()

    x = data['x-Coord']
    y = data['y-Coord']
    altitude = data['altitude']
    velocity = data['velocity']

    flight_params.init_flight_params(x, y, altitude, velocity)

    print(x)

    status = {"status": "OK"}

    return status
