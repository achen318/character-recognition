const canvas = document.getElementById('draw');
const ctx = canvas.getContext('2d');

// Add a white background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Round pen strokes of size 10
ctx.lineWidth = 10;
ctx.lineCap = 'round';

let penDown = false;

// Start drawing (mouse)
canvas.onmousedown = (e) => {
  penDown = true;
  ctx.beginPath();
};

// Start drawing (touch)
canvas.addEventListener(
  'touchstart',
  (e) => {
    e.preventDefault();

    penDown = true;
    ctx.beginPath();
  },
  { passive: false }
);

// Draw (mouse)
canvas.onmousemove = (e) => {
  if (penDown) {
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  }
};

// Draw (touch)
canvas.addEventListener(
  'touchmove',
  (e) => {
    e.preventDefault();

    if (penDown) {
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();

      ctx.lineTo(
        touch.pageX - rect.left - window.scrollX,
        touch.pageY - rect.top - window.scrollY
      );
      ctx.stroke();
    }
  },
  { passive: false }
);

// Stop drawing (mouse, touch)
canvas.onmouseup =
  canvas.onmouseleave =
  canvas.ontouchend =
  canvas.ontouchcancel =
    () => {
      penDown = false;
    };

// Reset canvas to white
document.getElementById('clear').onclick = () => {
  ctx.fillRect(0, 0, canvas.width, canvas.height);
};

// POST request, get predictions, reset canvas
document.getElementById('predict').onclick = () => {
  fetch('/predict', {
    method: 'POST',
    body: canvas.toDataURL().split(',')[1]
  })
    .then((res) => res.json())
    .then((data) => {
      document.getElementById('bg-pred').textContent = data['bg'];
      document.getElementById('ls-pred').textContent = data['ls'];
      document.getElementById('mm-pred').textContent = data['mm'];
      document.getElementById('mv-pred').textContent = data['mv'];
      document.getElementById('nn-pred').textContent = data['nn'];
    });

  ctx.fillRect(0, 0, canvas.width, canvas.height);
};
