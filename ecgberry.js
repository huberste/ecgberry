var BATCH_SIZE = 128; // data to be received and displayed at once
var SAMPLE_RATE = 853; // sample rate in samples per second
var MAX_DURATION = 10.0;
var MAX_FPS = 60.0;
var GRAPH_WIDTH = 600;

var GRAPH_ATTRS = {
    labels: ["t", "mV"],
    color: ["red"],
    digitsAfterDecimal: 3,
    width: 0.9 * document.documentElement.clientWidth,
    maxNumberWidth: 3,
    ylabel: "Voltage [mV]",
    xlabel: "Time [s]",
}

var fps = 5; // times per second the ECG will be graphed
var lastdraw = new Date().getTime();
var websocket;
var duration = 5.0; // graph length in seconds

var data1, data2, data3;
var annotations = [];
var pause = true; // if true, graph will not be updated
var ina1zero = 0;
var ina2zero = 0;
var ina3zero = 0;

var leads = 3;

var ecg1div, ecg2div, ecg3div, ecg4div, ecg5div, ecg6div, ecg7div;
var ecg1graph, ecg2graph, ecg3graph, ecg4graph, ecg5graph, ecg6graph, ecg7graph;

var chbxpause, chbxecg1display;
var btnconnect, btnstartthree, btnstartfive, btnstop, btndiagnose;
var inptfps, inptduration;
var divnumclients;
var divdiagnose;
var specg1label, specg2label, specg3label;

window.onload = function () {
    init();
}

function init() {
    // initialize
    chbxpause = document.getElementById("chbxpause");
    chbxecg1display = document.getElementById("chbxecg1display");
    pause = chbxpause.checked;
    btnconnect = document.getElementById("btnconnect");
    btnstartthree = document.getElementById("btnstartthree");
    btnstartfive = document.getElementById("btnstartfive");

    btnstop = document.getElementById("btnstop");
    btndiagnose = document.getElementById("btndiagnose");
    divnumclients = document.getElementById("divnumclients");
    divdiagnose = document.getElementById("divdiagnose");
    inptfps = document.getElementById("inptfps");
    inptduration = document.getElementById("inptduration");
    ecg1div = document.getElementById("divecg1");
    ecg2div = document.getElementById("divecg2");
    ecg3div = document.getElementById("divecg3");
    // ecg4div = document.getElementById("divecg4");
    // ecg5div = document.getElementById("divecg5");
    // ecg6div = document.getElementById("divecg6");
    // ecg7div = document.getElementById("divecg7");
    specg1label = document.getElementById("specg1label");
    specg2label = document.getElementById("specg2label");
    specg3label = document.getElementById("specg3label");
    // specg4label = document.getElementById("specg4label");
    // specg5label = document.getElementById("specg5label");
    // specg6label = document.getElementById("specg6label");
    // specg7label = document.getElementById("specg7label");

    data1 = generatedata();
    data2 = generatedata();
    data3 = generatedata();

    lastdraw = new Date().getTime();
    // print first graph
    ecg1graph = new Dygraph(
        ecg1div,
        data1,
        GRAPH_ATTRS
    );
    ecg2graph = new Dygraph(
        ecg2div,
        data2,
        GRAPH_ATTRS
    );
    ecg3graph = new Dygraph(
        ecg3div,
        data3,
        GRAPH_ATTRS
    );

    document.addEventListener("keydown", keyDown, false);

}

function keyDown(e) {
    var keyCode = e.keyCode;
    if(keyCode==32) {
        pause = !pause;
        chbxpause.checked = pause;
        e.preventDefault();
    }
}

function connect() {
    if (websocket) {
        websocket.close();
    }
    websocket = new WebSocket("ws://" + location.host + ":9669/");
    websocket.onmessage = function(event) {
        onmessage(event);
    }
    btnconnect.textContent = "(re)connect";
}

// when a websocket message comes in
function onmessage(event) {
    var data = JSON.parse(event.data);
    switch (data.type) {
        case 'data':
            process_batch(data);
            break;
        case 'status':
            process_status(data);
            break;
        case 'diagnose':
            process_diagnose(data);
            break;
        default:
            console.error(
                "unsupported event", data);
    }
};

function process_status(data) {
    btnstartthree.disabled = data.running;
    btnstartfive.disabled = data.running;
    btnstop.disabled = !data.running;
    btndiagnose.disabled = false;
    divnumclients.textContent = data.clients + " connected clients";
    ina1zero = data.ina1zero;
    ina2zero = data.ina2zero;
    ina3zero = data.ina3zero;
    leads = data.leads;
    if (leads == 3) {
        specg1label.textContent = "Lead I";
        specg2label.textContent = "Lead II";
        specg3label.textContent = "Lead III (calculated)";
    } else if (leads == 5) {
        specg1label.textContent = "Lead I";
        specg2label.textContent = "Lead II";
        specg3label.textContent = "Lead V";
    }
}

// process a batch of new data from the ecg
function process_batch(data) {
    // shift data
    for (var i = 0; i < ((SAMPLE_RATE * duration) - BATCH_SIZE); i++) {
        data1[i][1] = data1[i+BATCH_SIZE][1];
        data2[i][1] = data2[i+BATCH_SIZE][1];
        data3[i][1] = data3[i+BATCH_SIZE][1];
    }
    // insert new data
    for (var i = 0; i < BATCH_SIZE; i++) {
        data1[SAMPLE_RATE * duration - BATCH_SIZE + i][1] = data.data1[i];
        data2[SAMPLE_RATE * duration - BATCH_SIZE + i][1] = data.data2[i];
        data3[SAMPLE_RATE * duration - BATCH_SIZE + i][1] = data.data3[i];
    }
    for (var i = 0; i < annotations.length; i++) {
        // shift annotations by one batch size
        annotations[i].x = annotations[i].x - BATCH_SIZE;
        if (annotations[i].x < 0) {
            // delete this annotation
        }
    }
    offset = duration*SAMPLE_RATE-BATCH_SIZE;
    for (var i = 0; i < data.ps.length; i++) {
        annotations.push({
            series: "mV",
            x: offset + data.ps[i][1],
            shortText: "P"
        });
    }
    for (var i = 0; i < data.qs.length; i++) {
        annotations.push({
            series: "mV",
            x: offset + data.qs[i],
            shortText: "Q"
        });
    }
    for (var i = 0; i < data.rs.length; i++) {
        annotations.push({
            series: "mV",
            x: offset + data.rs[i],
            shortText: "R"
        });
    }
    for (var i = 0; i < data.ss.length; i++) {
        annotations.push({
            series: "mV",
            x: offset + data.ss[i],
            shortText: "S"
        });
    }
    for (var i = 0; i < data.ts.length; i++) {
        annotations.push({
            series: "mV",
            x: offset + data.ts[i][1],
            shortText: "T"
        });
    }
    if (pause) { return; } // Do not draw if PAUSE
    now = new Date().getTime() 
    if ((now - lastdraw) < (1000 / fps)) { return; }
    lastdraw = now
    // draw graph1
    if (ecg1graph) { ecg1graph.destroy(); }
    ecg1graph = new Dygraph(
        ecg1div,
        data1,
        GRAPH_ATTRS
    );
    // draw graph2
    if (ecg2graph) { ecg2graph.destroy(); }
    ecg2graph = new Dygraph(
        ecg2div,
        data2,
        GRAPH_ATTRS
    );
    ecg2graph.ready(function() {
        ecg2graph.setAnnotations(annotations);
    });
    // draw graph3
    if (ecg3graph) { ecg3graph.destroy(); }
    ecg3graph = new Dygraph(
        ecg3div,
        data3,
        GRAPH_ATTRS
    );
}

function process_diagnose(data) {
    //console.log(data)
    pause = true;
    chbxpause.checked = true;
    divdiagnose.textContent = data.diagnose;
    var num_samples = data.data1.length;
    duration = num_samples / SAMPLE_RATE;
    inptduration.value = Math.round(duration);
    data1 = [];
    data2 = [];
    data3 = [];
    data4 = [];
    data5 = [];
    data6 = [];
    data7 = [];
    displaydatas = [data1, data2, data3, data4, data5, data6, data7];
    ecgdatas = [data.data1, data.data2, data.data3, data.coeffs[0], data.coeffs[1], data.coeffs[2], data.coeffs[3], data.coeffs[4]];
    // console.log("heart rate:", data.heart_rate);
    var stepsize = 1 / SAMPLE_RATE;
    for (var j =0; j < 3; j++) {
        for (var i = 0; i < num_samples; i++) {
            var x = Math.round((-num_samples + i) * stepsize * 1000) / 1000
            displaydatas[j].push([i, ecgdatas[j][i]]);
        }
    }
    annotations = [];
    for (var i =0; i < data.r_peaks.length; i++) {
        var x = Math.round((-num_samples + data.r_peaks[i]) * stepsize * 1000)/1000;
        annotations.push({
            series: "mV",
            x: data.r_peaks[i],
            shortText: "R"
        });
    }
    for (var i = 0; i < data.onsets.length; i++) {
        var x = Math.round((-num_samples + data.onsets[i]) * stepsize * 1000)/1000;
        annotations.push({
            series: "mV",
            x: data.onsets[i],
            shortText: "Q"
        });
    }
    for (var i = 0; i < data.offsets.length; i++) {
        var x = Math.round((-num_samples + data.offsets[i]) * stepsize * 1000)/1000;
        annotations.push({
            series: "mV",
            x: data.offsets[i],
            shortText: "S"
        });
    }
    for (var i = 0; i < data.p_waves.length; i++) {
        var x = Math.round((-num_samples + data.p_waves[i][1]) * stepsize * 1000)/1000;
        annotations.push({
            series: "mV",
            x: data.p_waves[i][0],
            shortText: "<"
        });
        annotations.push({
            series: "mV",
            x: data.p_waves[i][1],
            shortText: "P"
        });
        annotations.push({
            series: "mV",
            x: data.p_waves[i][2    ],
            shortText: ">"
        });
    }
    for (var i = 0; i < data.t_waves.length; i++) {
        var x = Math.round((-num_samples + data.t_waves[i][1]) * stepsize * 1000) / 1000;
        annotations.push({
            series: "mV",
            x: data.t_waves[i][0],
            shortText: "<"
        });
        annotations.push({
            series: "mV",
            x: data.t_waves[i][1],
            shortText: "T"
        });
        annotations.push({
            series: "mV",
            x: data.t_waves[i][2],
            shortText: ">"
        });
    }
    // n_ks[0] displaying for debugging the algorithm
    // for (var i = 0; i < data.n_ks[0].length; i++) {
    //     var x = Math.round((-num_samples + data.n_ks[0][i]) * stepsize * 1000)/1000;
    //     annotations.push({
    //         series: "mV",
    //         x: data.n_ks[0][i],
    //         shortText: "1"
    //     });
    // }
    if (ecg1graph) { ecg1graph.destroy(); }
    ecg1graph = new Dygraph(
        ecg1div,
        data1,
        GRAPH_ATTRS
    );
    ecg1graph.ready(function() {
        ecg1graph.setAnnotations(annotations);
    });
    // draw graph2
    if (ecg2graph) { ecg2graph.destroy(); }
    ecg2graph = new Dygraph(
        ecg2div,
        data2,
        GRAPH_ATTRS
    );
    ecg2graph.ready(function() {
        ecg2graph.setAnnotations(annotations);
    });
    // draw graph3
    if (ecg3graph) { ecg3graph.destroy(); }
    ecg3graph = new Dygraph(
        ecg3div,
        data3,
        GRAPH_ATTRS
    );
    ecg3graph.ready(function() {
        ecg3graph.setAnnotations(annotations);
    });
    // if (ecg4graph) { ecg4graph.destroy(); }
    // ecg4graph = new Dygraph(
    //     ecg4div,
    //     data4,
    //     GRAPH_ATTRS
    // );
    // ecg4graph.ready(function() {
    //     ecg4graph.setAnnotations(annotations);
    // });
    // if (ecg5graph) { ecg5graph.destroy(); }
    // ecg5graph = new Dygraph(
    //     ecg5div,
    //     data5,
    //     GRAPH_ATTRS
    // );
    // ecg5graph.ready(function() {
    //     ecg5graph.setAnnotations(annotations);
    // });
    // if (ecg6graph) { ecg6graph.destroy(); }
    // ecg6graph = new Dygraph(
    //     ecg6div,
    //     data6,
    //     GRAPH_ATTRS
    // );
    // ecg6graph.ready(function() {
    //     ecg6graph.setAnnotations(annotations);
    // });
    // if (ecg7graph) { ecg7graph.destroy(); }
    // ecg7graph = new Dygraph(
    //     ecg7div,
    //     data7,
    //     GRAPH_ATTRS
    // );
    // ecg7graph.ready(function() {
    //     ecg7graph.setAnnotations(annotations);
    // });
}

function chbxpauseclick() {
    pause = chbxpause.checked;
}
function chbxecg1displayclick() {
    if (chbxecg1display.checked) {
        ecg1div.className = "";
    } else {
        ecg1div.className = "ghost";
    }
}

function btnstartthreeclick() {
    pause = false;
    chbxpause.checked = false;
    websocket.send(JSON.stringify({action: 'startthree'}));
}

function btnstartfiveclick() {
    pause = false;
    chbxpause.checked = false;
    websocket.send(JSON.stringify({action: 'startfive'}));
}

function btnstopclick() {
    websocket.send(JSON.stringify({action: 'stop'}));
}

function btndiagnoseclick() {
    pause = false;
    chbxpause.checked = false;
    websocket.send(JSON.stringify({action: 'diagnose'}));
}

function inptfpsinput() {
    if (inptfps.value <= MAX_FPS) {
        fps = inptfps.value;
    }
}

function inptdurationinput() {
    if (inptduration.value <= MAX_DURATION) {
        var duration_old = duration;
        duration = inptduration.value;
        if (duration_old > duration) {
            // remove values from front
            for (var i = 0; i < SAMPLE_RATE * (duration_old-duration); i++) {
                data1.shift();
                data2.shift();
                data3.shift();
            }
        } else if (duration_old < duration) {
            var stepsize = 1 / SAMPLE_RATE;
            // add values to front
             for (var i = 0; i < SAMPLE_RATE * (duration-duration_old); i++) {
        		x = Math.round(-duration_old - (i+1) * stepsize * 1000) / 1000
                data1.unshift([x, data1[0]]);
                data2.unshift([x, data2[0]]);
                data3.unshift([x, data3[0]]);
            }
        }
    }
}

function generatedata() {
    var result = [];
    var stepsize = 1 / SAMPLE_RATE;
    for (var i = 0; i < SAMPLE_RATE * duration; i++) {
        x = Math.round(-duration + i * stepsize * 1000) / 1000
        result.push([i, 0]);
    }
    return result;
}
