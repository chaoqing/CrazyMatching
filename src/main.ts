import './style.css'

const video = document.querySelector<HTMLVideoElement>('#video')!

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment' },
    audio: false,
  })
  video.srcObject = stream
}

setupCamera()
