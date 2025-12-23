// ==========================
// Emotion Recognition Logic
// ==========================

// Supported emotion classes (EXACT)
const EMOTIONS = [
	"Angry",
	"Disgust",
	"Fear",
	"Happy",
	"Neutral",
	"Sad",
	"Surprise",
];

const EMOJI_MAP = {
	Angry: "😠",
	Disgust: "🤢",
	Fear: "😨",
	Happy: "😄",
	Neutral: "😐",
	Sad: "😢",
	Surprise: "😲",
};

// DOM Elements
const floatingWindow = document.getElementById("floatingWindow");
const closeButton = document.getElementById("closeButton");

// Close button functionality
closeButton.addEventListener("click", () => {
	floatingWindow.style.transition = "all 0.5s ease-in-out";
	floatingWindow.style.transform = "translateY(-100%)";
	floatingWindow.style.opacity = "0";

	setTimeout(() => {
		floatingWindow.style.display = "none";
	}, 500);
});

const uploadArea = document.getElementById("uploadArea");
const imageInput = document.getElementById("imageInput");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const emotionLabel = document.getElementById("emotionLabel");
const emotionEmoji = document.getElementById("emotionEmoji");

// Upload handlers
uploadArea.addEventListener("click", () => imageInput.click());

uploadArea.addEventListener("dragover", (e) => {
	e.preventDefault();
	uploadArea.classList.add("hover");
});

uploadArea.addEventListener("dragleave", () => {
	uploadArea.classList.remove("hover");
});

uploadArea.addEventListener("drop", (e) => {
	e.preventDefault();
	uploadArea.classList.remove("hover");
	const file = e.dataTransfer.files[0];
	if (file) handleImage(file);
});

imageInput.addEventListener("change", () => {
	const file = imageInput.files[0];
	if (file) handleImage(file);
});

function handleImage(file) {
	const reader = new FileReader();
	reader.onload = () => {
		previewImage.src = reader.result;
		previewContainer.classList.remove("hidden");
		analyzeBtn.disabled = false;
		result.classList.add("hidden");
	};
	reader.readAsDataURL(file);
}

// ==========================
// Real Model Prediction Function
// ==========================
async function predictEmotion(imageElement) {
	try {
		// Get the image as base64
		const canvas = document.createElement("canvas");
		canvas.width = imageElement.naturalWidth;
		canvas.height = imageElement.naturalHeight;
		const ctx = canvas.getContext("2d");
		ctx.drawImage(imageElement, 0, 0);
		const imageData = canvas.toDataURL("image/jpeg");

		// Send to Flask API
		const response = await fetch("http://localhost:5000/predict", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ image: imageData }),
		});

		if (!response.ok) {
			throw new Error("Prediction failed");
		}

		const data = await response.json();
		console.log("Prediction results:", data); // للتحقق من النتائج
		return data.emotion;
	} catch (error) {
		console.error("Error:", error);
		alert("حدث خطأ في التحليل. تأكد من تشغيل Flask server");
		return "Neutral"; // Default emotion in case of error
	}
}

// Analyze button
analyzeBtn.addEventListener("click", async () => {
	loading.classList.remove("hidden");
	result.classList.add("hidden");

	const emotion = await predictEmotion(previewImage);
	const angryGifContainer = document.getElementById("angryGifContainer");
	const disgustGifContainer = document.getElementById("disgustGifContainer");
	const fearGifContainer = document.getElementById("fearGifContainer");
	const happyGifContainer = document.getElementById("happyGifContainer");
	const neutralGifContainer = document.getElementById("neutralGifContainer");
	const sadGifContainer = document.getElementById("sadGifContainer");
	const surpriseGifContainer = document.getElementById("surpriseGifContainer");

	loading.classList.add("hidden");

	emotionLabel.textContent = emotion;

	// Hide all GIF containers and emoji first
	emotionEmoji.classList.add("hidden");
	angryGifContainer.classList.add("hidden");
	disgustGifContainer.classList.add("hidden");
	fearGifContainer.classList.add("hidden");
	happyGifContainer.classList.add("hidden");
	neutralGifContainer.classList.add("hidden");
	sadGifContainer.classList.add("hidden");
	surpriseGifContainer.classList.add("hidden");

	// Handle different emotions
	if (emotion === "Angry") {
		angryGifContainer.classList.remove("hidden");
	} else if (emotion === "Disgust") {
		disgustGifContainer.classList.remove("hidden");
	} else if (emotion === "Fear") {
		fearGifContainer.classList.remove("hidden");
	} else if (emotion === "Happy") {
		happyGifContainer.classList.remove("hidden");
	} else if (emotion === "Neutral") {
		neutralGifContainer.classList.remove("hidden");
	} else if (emotion === "Sad") {
		sadGifContainer.classList.remove("hidden");
	} else if (emotion === "Surprise") {
		surpriseGifContainer.classList.remove("hidden");
	} else {
		emotionEmoji.textContent = EMOJI_MAP[emotion] || "❓";
		emotionEmoji.classList.remove("hidden");
	}

	result.classList.remove("hidden");
});
