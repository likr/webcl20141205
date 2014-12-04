$(function() {
  if (window.webcl === undefined) {
    alert('えっ、まだWebCL未対応ブラウザ使ってるの? 信じられない...');
    return false;
  }

  var kernelSource;
  var devices = getDevices();

  devices.forEach(function(d, i) {
    var platformName = d[0].getInfo(WebCL.PLATFORM_NAME);
    var deviceName = d[1].getInfo(WebCL.DEVICE_NAME);
    var label = platformName + ' - ' + deviceName;
    $('#devices').append($('<option>').html(label).val(i));
  });
  $('#devices').append($('<option>').html('JS').val(-1));

  $('#run-button').click(function() {
    var select = $('#devices').val();
    var blockSize = +$('#block-size').val();
    var scale = +$('#scale').val();
    var repeat = +$('#repeat').val();
    var size = blockSize * scale;

    var A = new Float32Array(size * size);
    var B = new Float32Array(size * size);
    var C = new Float32Array(size * size);
    var i, j;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        A[i * size + j] = Math.random() * 10;
        B[i * size + j] = Math.random() * 10;
      }
    }

    var matMul;
    if (select < 0) {
      matMul = matMulJS;
    } else {
      var device = devices[select][1];
      matMul = genMatMulCL(device, blockSize);
    }

    var time = getTime(repeat, function() {
      matMul(A, B, C, size, size, size);
    });

    var tr = $('<tr>');
    tr.append($('<td>').html(select < 0 ? 'JS' : $('#devices option:selected').text()));
    tr.append($('<td>').html(blockSize));
    tr.append($('<td>').html(scale));
    tr.append($('<td>').html(time));
    $('#result').append(tr);
    console.log(C);
  });

  $.get('matrixMul.cl')
    .done(function(src) {
      kernelSource = src;
      $('#run-button').attr('disabled', false);
    });

  function getDevices() {
    var result = [];
    webcl.getPlatforms().forEach(function(platform) {
      platform.getDevices().forEach(function(device) {
        result.push([platform, device]);
      });
    });
    return result;
  }

  function matMulJS(A, B, C, wa, ha, hb) {
    var i, j, k;
    for (i = 0; i < wa; ++i) {
      for (j = 0; j < hb; ++j) {
        var val = 0;
        var iA = wa * i;
        var iB = j;
        for (k = 0; k < ha; ++k) {
          val += A[iA] * B[iB];
          iA += 1;
          iB += ha;
        }
        C[i * wa + j] = val;
      }
    }
  }

  function genMatMulCL(device, blockSize) {
    var context = webcl.createContext(device);
    var program = context.createProgram(kernelSource);
    program.build([device], '-D BLOCK_SIZE=' + blockSize);
    var kernel = program.createKernel('matrixMul');
    var queue = context.createCommandQueue(device);
    var floatSize = 4;

    return function matMulCL(A, B, C, wa, ha, hb) {
      var devA = context.createBuffer(WebCL.MEM_READ_ONLY, floatSize * wa * ha);
      var devB = context.createBuffer(WebCL.MEM_READ_ONLY, floatSize * ha * hb);
      var devC = context.createBuffer(WebCL.MEM_WRITE_ONLY, floatSize * wa * hb);

      kernel.setArg(0, devC);
      kernel.setArg(1, devA);
      kernel.setArg(2, devB);
      kernel.setArg(3, new Uint32Array([floatSize * blockSize * blockSize]));
      kernel.setArg(4, new Uint32Array([floatSize * blockSize * blockSize]));
      kernel.setArg(5, new Uint32Array([wa]));
      kernel.setArg(6, new Uint32Array([hb]));
      kernel.setArg(7, new Uint32Array([ha]));

      queue.enqueueWriteBuffer(devA, false, 0, floatSize * wa * ha, A);
      queue.enqueueWriteBuffer(devB, false, 0, floatSize * ha * hb, B);

      queue.enqueueNDRangeKernel(kernel, 2, null, [wa, hb], [blockSize, blockSize]);
      queue.enqueueReadBuffer(devC, false, 0, floatSize * wa * hb, C);
      queue.finish();
    };
  }

  function getTime(times, f) {
    var i;
    var start = new Date();
    for (i = 0; i < times; ++i) {
      f();
    }
    var stop = new Date();
    return (stop - start) / times;
  }
});
