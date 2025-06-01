let featuresChart = null;
document
.getElementById("singleImageInput")
.addEventListener("change", function (e) {
    const file = e.target.files[0];
    const previewImage = document.getElementById("previewImage");
    const tiffWarning = document.getElementById("tiffWarning");
    const canvas = document.getElementById("tiffCanvas");
    const ctx = canvas.getContext("2d");

    if (!file) return;

    const fileType = file.type.toLowerCase();
    if (fileType === "image/tiff" || file.name.endsWith(".tif")) {
    tiffWarning.innerHTML = "";

    const reader = new FileReader();
    reader.onload = function (e) {
        const arrayBuffer = e.target.result;
        const ifds = UTIF.decode(arrayBuffer);
        UTIF.decodeImage(arrayBuffer, ifds[0]);
        const rgba = UTIF.toRGBA8(ifds[0]);

        canvas.width = ifds[0].width;
        canvas.height = ifds[0].height;
        const imageData = ctx.createImageData(
        canvas.width,
        canvas.height
        );
        imageData.data.set(rgba);
        ctx.putImageData(imageData, 0, 0);

        previewImage.style.display = "none";
        canvas.style.display = "block";
    };
    reader.readAsArrayBuffer(file);
    } else {
    tiffWarning.innerHTML = "";
    const reader = new FileReader();
    reader.onload = function (e) {
        previewImage.src = e.target.result;
        previewImage.classList.add("active");
        previewImage.style.display = "block";
        canvas.style.display = "none";
    };
    reader.readAsDataURL(file);
    file.value = "";

    }
});
document.getElementById("folderImageInput").addEventListener("change", function (e) {
const files = e.target.files;
const previewContainer = document.getElementById("previewContainer");
previewContainer.innerHTML = "";

for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const fileType = file.type.toLowerCase();

    if (fileType === "image/tiff" || file.name.endsWith(".tif")) {
    const reader = new FileReader();
    reader.onload = function (e) {
        const arrayBuffer = e.target.result;
        const ifds = UTIF.decode(arrayBuffer);
        UTIF.decodeImage(arrayBuffer, ifds[0]);
        const rgba = UTIF.toRGBA8(ifds[0]);

        const canvas = document.createElement("canvas");
        canvas.width = ifds[0].width;
        canvas.height = ifds[0].height;
        const ctx = canvas.getContext("2d");
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        imageData.data.set(rgba);
        ctx.putImageData(imageData, 0, 0);

        previewContainer.appendChild(canvas);
    };
    reader.readAsArrayBuffer(file);
    } else {
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = document.createElement("img");
        img.src = e.target.result;
        img.style.maxWidth = "100%";
        img.style.marginBottom = "1rem";
        img.style.border = "1px solid #ccc";
        img.style.borderRadius = "8px";
        previewContainer.appendChild(img);
    };
    reader.readAsDataURL(file);
    }
}
});
async function uploadImageSingle() {

const fileInput = document.getElementById("singleImageInput");
const resultDiv = document.getElementById("result");
const loadingDiv = document.getElementById("loading");

if (!fileInput.files[0]) {
    alert("Vui lòng chọn ảnh trước khi phân tích!");
    return;
}

const formData = new FormData();
formData.append("file", fileInput.files[0]);

try {
    loadingDiv.style.display = "block";
    resultDiv.innerHTML = "";
    

    const response = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData,
    });

    if (!response.ok)
    throw new Error(`HTTP error! status: ${response.status}`);

    const data = await response.json();
    const features = data.features;

    resultDiv.innerHTML = `
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <p><strong>📄 Tên file:</strong> ${data.image}</p>
        <p><strong>🎯 Kết quả:</strong> <span style="color: #27ae60;">${data.predicted_class}</span></p>

        <hr>

        <h5>📊 Đặc trưng ảnh:</h5>
        <ul>
        <li><strong>Màu sắc:</strong> [${features.color.slice(0, 5).join(', ')} ...]</li>
        <li><strong>GLCM:</strong> [${features.glcm.slice(0, 5).join(', ')} ...]</li>
        <li><strong>LBP:</strong> [${features.lbp.slice(0, 5).join(', ')} ...]</li>
        <li><strong>GIST:</strong> [${features.gist.slice(0, 5).join(', ')} ...]</li>
        </ul>

        <hr>

        <h5>🖼️ Hình ảnh và Biểu đồ đặc trưng:</h5>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
        <div>
            <img src="data:image/jpeg;base64,${data.image_base64}" alt="Original Image" style="max-width: 100%; border: 1px solid #ccc; padding: 5px;" />
        </div>
        <div>
            <img src="data:image/png;base64,${data.chart_base64}" alt="Feature Chart" style="max-width: 100%; border: 1px solid #ccc; padding: 5px;" />
        </div>
        </div>
    </div>
    `;
    loadingDiv.style.display = "none";
    resultDiv.style.display = "block";
    
    

    // Hiển thị biểu đồ đặc trưng ảnh
    const chartData = {
    labels: ['SIFT','Màu sắc', 'GLCM', 'LBP', 'GIST'],
    datasets: [{
        label: 'Đặc trưng ảnh',
        data: [
        features.sift.length,
        features.color.length, 
        features.glcm.length, 
        features.lbp.length, 
        features.gist.length
        ], 
        backgroundColor: [
        'rgba(128, 0, 128, 0.2)',
        'rgba(41, 128, 185, 0.2)', 
        'rgba(46, 204, 113, 0.2)', 
        'rgba(241, 196, 15, 0.2)', 
        'rgba(231, 76, 60, 0.2)'
        ],
        borderColor: [
        'rgba(128, 0, 128, 0.2)', 
        'rgba(41, 128, 185, 1)', 
        'rgba(46, 204, 113, 1)', 
        'rgba(241, 196, 15, 1)', 
        'rgba(231, 76, 60, 1)'
        ],
        borderWidth: 1
    }]
    };
    document.getElementById("chartSection").style.display = "block";
    const ctx = document.getElementById('featuresChart').getContext('2d');
    if (featuresChart) {
    featuresChart.destroy();
    featuresChart = null;
    }
    featuresChart = new Chart(ctx, {
    type: 'bar', // Loại biểu đồ
    data: chartData,
    options: {
        responsive: true,
        scales: {
        y: {
            beginAtZero: true
        }
        }
    }
    });
    
    } catch (error) {
    resultDiv.innerHTML = `<p style="color: #e74c3c;">❌ Lỗi: ${error.message}</p>`;
    console.error("Lỗi khi gửi ảnh:", error);
    } finally {
    loadingDiv.style.display = "none";
    }
}
async function uploadImage() {
const fileInput = document.getElementById("folderImageInput");
const folderNameInput = document.getElementById("folderNameInput");
const resultDiv = document.getElementById("result");
const loadingDiv = document.getElementById("loading");
const downloadLink = document.getElementById("downloadLink");

const files = fileInput.files;
const folderName = folderNameInput.value.trim();

resultDiv.innerHTML = "";
downloadLink.innerHTML = "";
// document.getElementById("chartSection").style.display = "none";

if (!folderName || !files.length) {
    alert("Vui lòng nhập tên thư mục và chọn ảnh trước khi phân tích!");
    return;
}


const formData = new FormData();
formData.append("folder_name", folderName);
for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
}

loadingDiv.style.display = "block";
resultDiv.style.display = "block";

try {
    const response = await fetch("http://127.0.0.1:5000/predict-folder", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) throw new Error(`Lỗi ${response.status}`);

    const data = await response.json();
    const results = data.results;
    for (let i = 0; i < results.length; i++) {
        const data = results[i];
        

    resultDiv.innerHTML += `
        <div  class="result-card" style="display: block; margin-bottom: 20px;">
        <p><strong>📄 Tên file:</strong> ${data.image}</p>
        <p><strong>🎯 Kết quả:</strong> <span style="color: #27ae60;">${data.predicted_class}</span></p>

        <hr>

        <h5>📊 Đặc trưng ảnh:</h5>
        <ul>
            <li><strong>Màu sắc:</strong> [${data.features.color.slice(0, 5).join(', ')} ...]</li>
            <li><strong>GLCM:</strong> [${data.features.glcm.slice(0, 5).join(', ')} ...]</li>
            <li><strong>LBP:</strong> [${data.features.lbp.slice(0, 5).join(', ')} ...]</li>
            <li><strong>GIST:</strong> [${data.features.gist.slice(0, 5).join(', ')} ...]</li>
        </ul>

        <hr>

        <h5>🖼️ Hình ảnh và Biểu đồ đặc trưng:</h5>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div>
            <img src="data:image/jpeg;base64,${data.image_base64}" alt="Original Image" style="max-width: 100%; border: 1px solid #ccc; padding: 5px;" />
            </div>
            <div>
            <img src="data:image/png;base64,${data.chart_base64}" alt="Feature Chart" style="max-width: 100%; border: 1px solid #ccc; padding: 5px;" />
            </div>
        </div>
        </div>
    `;
    loadingDiv.style.display = "none";
    const chartData = {
        labels: ["SIFT", "Màu sắc", "GLCM", "LBP", "GIST"],
        datasets: [{
        label: "Đặc trưng ảnh",
        data: [
        data.features.sift.length,
        data.features.color.length,
        data.features.glcm.length,
        data.features.lbp.length,
        data.features.gist.length,
        ],
        backgroundColor: [
            "rgba(128, 0, 128, 0.2)",
            "rgba(41, 128, 185, 0.2)",
            "rgba(46, 204, 113, 0.2)",
            "rgba(241, 196, 15, 0.2)",
            "rgba(231, 76, 60, 0.2)",
        ],
        borderColor: [
            "rgba(128, 0, 128, 1)",
            "rgba(41, 128, 185, 1)",
            "rgba(46, 204, 113, 1)",
            "rgba(241, 196, 15, 1)",
            "rgba(231, 76, 60, 1)",
        ],
        borderWidth: 1,
        }],
    };

    
    const ctx = document.getElementById("featuresChart").getContext("2d");
    if (featuresChart) {
        featuresChart.destroy();
        featuresChart = null;
    }
    featuresChart = new Chart(ctx, {
        type: "bar",
        data: chartData,
        options: {
        responsive: true,
        scales: {
            y: {
            beginAtZero: true,
            },
        },
        },
    });

    document.getElementById("chartSection").style.display = "block";
    }

    // Hiện liên kết tải CSV
    downloadLink.innerHTML = `
        <a href="http://127.0.0.1:5000/download-results/${data.folder_id}" class="download-btn" target="_blank">
        ⬇️ Tải kết quả CSV
        </a>
    `;
} catch (err) {
    resultDiv.innerHTML = `<div class="result-card" style="border-left: 4px solid red;">
    <p>❌ Lỗi gửi thư mục: ${err.message}</p>
    </div>`;
} finally {
    loadingDiv.style.display = "none";
}
}