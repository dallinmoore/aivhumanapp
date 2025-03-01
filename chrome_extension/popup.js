// New function to handle file upload
async function uploadFile(file) {
    if (!file) {
        alert("Please select an image.");
        return;
    }
    // Update the drop zone text to indicate file selection
    document.getElementById("dropZone").innerText = "File uploaded";
    
    let formData = new FormData();
    formData.append("file", file);

    try {
        // Use your machine's IP if 127.0.0.1 isn't resolving from the extension:
        let response = await fetch("http://127.0.0.1:5000/api/upload", {
            method: "POST",
            body: formData
        });
        let data = await response.json();
        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
        } else {
            document.getElementById("result").innerText = `Prediction: ${data.prediction} (Confidence: ${data.confidence}%)`;
        }
    } catch (error) {
        console.error("Upload error:", error);
        document.getElementById("result").innerText = "Failed to connect to the server.";
    }
}

// Handle click on Analyze button
document.getElementById("uploadBtn").addEventListener("click", async function() {
    let file = document.getElementById("fileInput").files[0];
    uploadFile(file);
});

// Trigger file input when drop zone is clicked
document.getElementById("dropZone").addEventListener("click", function() {
    document.getElementById("fileInput").click();
});

// Automatically upload when a file is selected from the file input
document.getElementById("fileInput").addEventListener("change", function(event) {
    let file = event.target.files[0];
    if(file) {
        uploadFile(file);
    }
});

// Drag & Drop events for drop zone
let dropZone = document.getElementById("dropZone");

dropZone.addEventListener("dragover", function(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.borderColor = "green";
});

dropZone.addEventListener("dragleave", function(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.borderColor = "#ccc";
});

dropZone.addEventListener("drop", async function(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.borderColor = "#ccc";
    let files = event.dataTransfer.files;
    if (files.length > 0) {
        // Update hidden file input and upload the file
        document.getElementById("fileInput").files = files;
        uploadFile(files[0]);
    }
});
