window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/pose_den";
var INTERP_BASE_1 = "./static/interpolation/pose_den_1";
var INTERP_BASE_2 = "./static/interpolation/ik";


var NUM_INTERP_FRAMES = 10;
var NUM_INTERP_FRAMES_1 = 10;
var NUM_INTERP_FRAMES_2 = 11;



var interp_images = [];
var interp_images_1 = [];
var interp_images_2 = [];



function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    // var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.png';
    var path = INTERP_BASE + '/img_' + String(i) + '.png';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function preloadInterpolationImages_1() {
  for (var i = 0; i < NUM_INTERP_FRAMES_1; i++) {
    var path = INTERP_BASE_1 + '/img_' + String(i) + '.png';
    interp_images_1[i] = new Image();
    interp_images_1[i].src = path;
  }
}

function preloadInterpolationImages_2() {
  for (var i = 0; i < NUM_INTERP_FRAMES_2; i++) {
    var path = INTERP_BASE_2 + '/img_' + String(i) + '.png';
    interp_images_2[i] = new Image();
    interp_images_2[i].src = path;
  }
}



function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

function setInterpolationImage_1(i) {
  var image = interp_images_1[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-1').empty().append(image);
}

function setInterpolationImage_2(i) {
  var image = interp_images_2[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-2').empty().append(image);
}



$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

    preloadInterpolationImages_1();

    $('#interpolation-slider-1').on('input', function(event) {
      setInterpolationImage_1(this.value);
    });
    setInterpolationImage_1(0);
    $('#interpolation-slider-1').prop('max', NUM_INTERP_FRAMES_1 - 1);

    bulmaSlider.attach();

    preloadInterpolationImages_2();

    $('#interpolation-slider-2').on('input', function(event) {
      setInterpolationImage_2(this.value);
    });
    setInterpolationImage_2(0);
    $('#interpolation-slider-2').prop('max', NUM_INTERP_FRAMES_2 - 1);
    
    bulmaSlider.attach();
    


})