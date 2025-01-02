
"use strict";

let Gpgst = require('./Gpgst.js');
let Sentence = require('./Sentence.js');
let Gpgsa = require('./Gpgsa.js');
let GpgsvSatellite = require('./GpgsvSatellite.js');
let Gprmc = require('./Gprmc.js');
let Gpgsv = require('./Gpgsv.js');
let Gpgga = require('./Gpgga.js');

module.exports = {
  Gpgst: Gpgst,
  Sentence: Sentence,
  Gpgsa: Gpgsa,
  GpgsvSatellite: GpgsvSatellite,
  Gprmc: Gprmc,
  Gpgsv: Gpgsv,
  Gpgga: Gpgga,
};
