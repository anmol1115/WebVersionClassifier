let canvas = document.getElementById('myCanvas')
let clearCanvas = document.getElementById('clearCanvas')
let predictBtn = document.getElementById('predictBtn')
let base64Image
const canvasSide = 300
let context = canvas.getContext('2d')

canvas.width = canvasSide
canvas.height = canvasSide
let painting = false

function startPosition(e) {
    painting = true
    draw(e)
}

function endPosition() {
    painting = false
    context.beginPath()
}

function draw(e) {
    if (!painting) {
        return
    }
    let position = canvas.getBoundingClientRect()
    context.lineWidth = 20
    context.lineCap = "round"

    let x = position.left
    let y = position.top
    context.lineTo(e.clientX - x, e.clientY - y)
    context.stroke()

    context.beginPath()
    context.moveTo(e.clientX - x, e.clientY - y)
}

function submit() {
    let image = new Image()
    image.src = canvas.toDataURL()
    base64Image = image.src.replace('data:image/png;base64,', '')
    // console.log(base64Image)
    let message = {
        image: base64Image
    }
    $.post('http://127.0.0.1:5000/home', JSON.stringify(message), function (obj) {
        JSC.Chart('chartDiv', {
            type: 'vertical column',
            series: [
                {
                    points: [
                        {x:'zero', y:obj['zero']},
                        {x:'one', y:obj['one']},
                        {x:'two', y:obj['two']},
                        {x:'three', y:obj['three']},
                        {x:'four', y:obj['four']},
                        {x:'five', y:obj['five']},
                        {x:'six', y:obj['six']},
                        {x:'seven', y:obj['seven']},
                        {x:'eight', y:obj['eight']},
                        {x:'nine', y:obj['nine']}
                    ]
                }
            ]
        })
    })
}

function erase() {
    context.clearRect(0, 0, canvas.width, canvas.height)
}

canvas.addEventListener('mousedown', startPosition)
canvas.addEventListener('mouseup', endPosition)
canvas.addEventListener('mousemove', draw)
clearCanvas.addEventListener('click', erase)
predictBtn.addEventListener('click', submit)