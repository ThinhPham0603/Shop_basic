// const { spawn } = require("child_process");
// const path = require("path");
// const fs = require("fs");
// const forge = require("node-forge");

// const keyDir = path.join(__dirname, "../keys");

// if (!fs.existsSync(keyDir)) {
//   fs.mkdirSync(keyDir, { recursive: true });
// }

// const privateKeyPath = path.join(keyDir, "private.pem");
// const publicKeyPath = path.join(keyDir, "public.pem");

// let publicKey, privateKey;

// if (fs.existsSync(privateKeyPath) && fs.existsSync(publicKeyPath)) {
//   privateKey = fs.readFileSync(privateKeyPath, "utf8");
//   publicKey = fs.readFileSync(publicKeyPath, "utf8");
// } else {
//   const keys = forge.pki.rsa.generateKeyPair({ bits: 2048 });
//   publicKey = forge.pki.publicKeyToPem(keys.publicKey);
//   privateKey = forge.pki.privateKeyToPem(keys.privateKey);

//   fs.writeFileSync(privateKeyPath, privateKey);
//   fs.writeFileSync(publicKeyPath, publicKey);
// }

// const encryptWithRSA = (text, publicKey) => {
//   try {
//     const pubKey = forge.pki.publicKeyFromPem(publicKey);
//     const encrypted = pubKey.encrypt(forge.util.encodeUtf8(text), "RSA-OAEP");
//     return forge.util.encode64(encrypted);
//   } catch (error) {
//     console.error("Error encrypting data:", error);
//     return null;
//   }
// };

// const decryptWithRSA = (encryptedText, privateKey) => {
//   try {
//     const privKey = forge.pki.privateKeyFromPem(privateKey);
//     const decrypted = privKey.decrypt(
//       forge.util.decode64(encryptedText),
//       "RSA-OAEP"
//     );
//     return forge.util.decodeUtf8(decrypted);
//   } catch (error) {
//     console.error("Error decrypting data:", error);
//     return null;
//   }
// };

// const recognizeSpeech = async (req, res) => {
//   try {
//     if (!req.file || !req.file.buffer) {
//       return res
//         .status(400)
//         .json({ success: false, message: "No audio file uploaded" });
//     }

//     const audio = req.file.buffer;
//     const webmFilePath = path.join(__dirname, `../uploads/${Date.now()}.webm`);
//     fs.writeFileSync(webmFilePath, audio);

//     const pythonScriptPath = path.join(
//       __dirname,
//       "../scripts/process_speech.py"
//     );
//     const pythonProcess = spawn("python", [pythonScriptPath, webmFilePath]);

//     let transcription = "";
//     let errorLogs = "";

//     pythonProcess.stdout.on("data", (data) => {
//       transcription += data.toString();
//     });

//     pythonProcess.stderr.on("data", (data) => {
//       const errorMessage = data.toString();
//       if (!errorMessage.includes("FP16 is not supported on CPU")) {
//         errorLogs += errorMessage;
//         console.error(`Python error: ${errorMessage}`);
//       }
//     });

//     pythonProcess.on("close", (code) => {
//       console.log(`Python process exited with code ${code}`);

//       if (code === 0 && transcription.trim()) {
//         const encryptedData = encryptWithRSA(transcription.trim(), publicKey);
//         if (!encryptedData) {
//           return res
//             .status(500)
//             .json({ success: false, message: "Encryption failed" });
//         }
//         return res.json({ success: true, encryptedData });
//       } else if (code === 0 && !transcription.trim()) {
//         return res.json({
//           success: false,
//           message: "Không nhận dạng được giọng nói",
//         });
//       } else {
//         return res.status(500).json({
//           success: false,
//           message: "Speech recognition failed",
//           details: errorLogs || "Unknown error",
//         });
//       }
//     });
//   } catch (error) {
//     console.error("Error processing audio:", error);
//     if (!res.headersSent) {
//       res.status(500).json({
//         success: false,
//         message: `Error processing audio: ${error.message}`,
//       });
//     }
//   }
// };

// const decryptText = async (req, res) => {
//   try {
//     const { encryptedText } = req.body;
//     if (!encryptedText) {
//       return res.status(400).json({ error: "Thiếu dữ liệu để giải mã." });
//     }

//     const decryptedText = decryptWithRSA(encryptedText, privateKey);
//     if (!decryptedText) {
//       return res.status(500).json({ error: "Giải mã thất bại." });
//     }

//     res.json({ decryptedText });
//   } catch (error) {
//     console.error("Lỗi giải mã:", error);
//     res.status(500).json({ error: "Lỗi trong quá trình giải mã." });
//   }
// };

// module.exports = { recognizeSpeech, decryptText };

// const { spawn } = require("child_process");
// const path = require("path");
// const fs = require("fs");
// const forge = require("node-forge");
// const crypto = require("crypto");

// const keyDir = path.join(__dirname, "../keys");
// if (!fs.existsSync(keyDir)) {
//   fs.mkdirSync(keyDir, { recursive: true });
// }

// const privateKeyPath = path.join(keyDir, "private.pem");
// const publicKeyPath = path.join(keyDir, "public.pem");

// let publicKey, privateKey;
// if (fs.existsSync(privateKeyPath) && fs.existsSync(publicKeyPath)) {
//   privateKey = fs.readFileSync(privateKeyPath, "utf8");
//   publicKey = fs.readFileSync(publicKeyPath, "utf8");
// } else {
//   const keys = forge.pki.rsa.generateKeyPair({ bits: 2048 });
//   publicKey = forge.pki.publicKeyToPem(keys.publicKey);
//   privateKey = forge.pki.privateKeyToPem(keys.privateKey);
//   fs.writeFileSync(privateKeyPath, privateKey);
//   fs.writeFileSync(publicKeyPath, publicKey);
// }
// function checkKeySize(privateKey) {
//   try {
//     const keyObject = crypto.createPrivateKey(privateKey);
//     const keySize = keyObject.asymmetricKeyDetails.modulusLength;
//     console.log(`🔍 Kích thước khóa: ${keySize} bit`);
//     if (keySize >= 2048) {
//       console.log("✅ Khóa đạt tiêu chuẩn NIST (>= 2048-bit)");
//       return true;
//     } else {
//       console.log("❌ Khóa không đạt tiêu chuẩn NIST!");
//       return false;
//     }
//   } catch (error) {
//     console.error("❌ Lỗi khi kiểm tra độ dài khóa:", error.message);
//   }
// }

// function checkPublicExponent(publicKey) {
//   try {
//     const keyObject = crypto.createPublicKey(publicKey);
//     const exponent = keyObject.asymmetricKeyDetails.publicExponent;
//     console.log(`🔍 Số mũ công khai: ${exponent}`);
//     if (Number(exponent) === 65537) {
//       console.log("✅ Số mũ công khai đạt tiêu chuẩn NIST (65537)");
//       return true;
//     } else {
//       console.log("❌ Số mũ công khai không đạt tiêu chuẩn!");
//       return false;
//     }
//   } catch (error) {
//     console.error("❌ Lỗi khi kiểm tra số mũ công khai:", error.message);
//   }
// }

// function checkEntropy(privateKey) {
//   try {
//     const hash = crypto.createHash("sha256").update(privateKey).digest("hex");
//     console.log(`🔍 SHA-256 của khóa: ${hash}`);
//     console.log(
//       "✅ Nếu SHA-256 không có mẫu lặp đáng ngờ, khóa có độ ngẫu nhiên tốt."
//     );
//     return true;
//   } catch (error) {
//     console.error("❌ Lỗi khi kiểm tra entropy:", error.message);
//   }
// }

// (function validateRSAKeys() {
//   console.log("\n--- 🔍 Kiểm tra khóa RSA theo tiêu chuẩn NIST ---");

//   const keySizeValid = checkKeySize(privateKey);
//   const exponentValid = checkPublicExponent(publicKey);
//   const entropyValid = checkEntropy(privateKey);

//   if (keySizeValid && exponentValid && entropyValid) {
//     console.log("✅ Đã vượt qua tiêu chuẩn NIST");
//     return true;
//   } else {
//     console.log("❌ Chưa đạt tiêu chuẩn NIST!");
//     return false;
//   }
// })();

// const generateAESKey = () => forge.random.getBytesSync(32);

// const encryptWithAES = (text, aesKey) => {
//   try {
//     const iv = forge.random.getBytesSync(16);
//     const cipher = forge.cipher.createCipher("AES-CBC", aesKey);
//     cipher.start({ iv });
//     cipher.update(forge.util.createBuffer(text, "utf8"));
//     cipher.finish();
//     return {
//       encryptedData: forge.util.encode64(cipher.output.getBytes()),
//       iv: forge.util.encode64(iv),
//     };
//   } catch (error) {
//     console.error("Error encrypting with AES:", error);
//     return null;
//   }
// };

// const encryptAESKeyWithRSA = (aesKey, publicKey) => {
//   try {
//     const pubKey = forge.pki.publicKeyFromPem(publicKey);
//     return forge.util.encode64(pubKey.encrypt(aesKey, "RSA-OAEP"));
//   } catch (error) {
//     console.error("Error encrypting AES key with RSA:", error);
//     return null;
//   }
// };

// const encryptData = (text, publicKey) => {
//   const aesKey = generateAESKey();
//   const encryptedAESKey = encryptAESKeyWithRSA(aesKey, publicKey);
//   if (!encryptedAESKey) return null;
//   const { encryptedData, iv } = encryptWithAES(text, aesKey);
//   return { encryptedData, iv, encryptedAESKey };
// };

// const decryptAESKeyWithRSA = (encryptedKey, privateKey) => {
//   try {
//     const privKey = forge.pki.privateKeyFromPem(privateKey);
//     return privKey.decrypt(forge.util.decode64(encryptedKey), "RSA-OAEP");
//   } catch (error) {
//     console.error("Error decrypting AES key with RSA:", error);
//     return null;
//   }
// };

// const decryptWithAES = (encryptedData, aesKey, iv) => {
//   try {
//     const decipher = forge.cipher.createDecipher("AES-CBC", aesKey);
//     decipher.start({ iv: forge.util.decode64(iv) });
//     decipher.update(
//       forge.util.createBuffer(forge.util.decode64(encryptedData))
//     );
//     decipher.finish();
//     return decipher.output.toString("utf8");
//   } catch (error) {
//     console.error("Error decrypting with AES:", error);
//     return null;
//   }
// };

// const decryptData = (encryptedData, iv, encryptedAESKey, privateKey) => {
//   const aesKey = decryptAESKeyWithRSA(encryptedAESKey, privateKey);
//   if (!aesKey) return null;
//   return decryptWithAES(encryptedData, aesKey, iv);
// };

// const recognizeSpeech = async (req, res) => {
//   try {
//     if (!req.file || !req.file.buffer) {
//       return res
//         .status(400)
//         .json({ success: false, message: "No audio file uploaded" });
//     }

//     const audio = req.file.buffer;
//     const webmFilePath = path.join(__dirname, `../uploads/${Date.now()}.webm`);
//     fs.writeFileSync(webmFilePath, audio);

//     const pythonScriptPath = path.join(
//       __dirname,
//       "../scripts/process_speech.py"
//     );
//     const pythonProcess = spawn("python", [pythonScriptPath, webmFilePath]);

//     let transcription = "";
//     let errorLogs = "";

//     pythonProcess.stdout.on("data", (data) => {
//       transcription += data.toString();
//     });

//     pythonProcess.stderr.on("data", (data) => {
//       const errorMessage = data.toString();
//       if (!errorMessage.includes("FP16 is not supported on CPU")) {
//         errorLogs += errorMessage;
//         console.error(`Python error: ${errorMessage}`);
//       }
//     });

//     pythonProcess.on("close", (code) => {
//       console.log(`Python process exited with code ${code}`);

//       if (code === 0 && transcription.trim()) {
//         const encryptedData = encryptData(transcription.trim(), publicKey);
//         if (!encryptedData) {
//           return res
//             .status(500)
//             .json({ success: false, message: "Encryption failed" });
//         }
//         return res.json({ success: true, encryptedData });
//       } else if (code === 0 && !transcription.trim()) {
//         return res.json({
//           success: false,
//           message: "Không nhận dạng được giọng nói",
//         });
//       } else {
//         return res.status(500).json({
//           success: false,
//           message: "Speech recognition failed",
//           details: errorLogs || "Unknown error",
//         });
//       }
//     });
//   } catch (error) {
//     console.error("Error processing audio:", error);
//     if (!res.headersSent) {
//       res.status(500).json({
//         success: false,
//         message: `Error processing audio: ${error.message}`,
//       });
//     }
//   }
// };

// const decryptText = async (req, res) => {
//   try {
//     const { encryptedData, iv, encryptedAESKey } = req.body.encryptedTranscript;
//     const decryptedText = decryptData(
//       encryptedData,
//       iv,
//       encryptedAESKey,
//       privateKey
//     );
//     if (!decryptedText) {
//       return res.status(500).json({ error: "Decryption failed." });
//     }
//     res.json({ decryptedText });
//   } catch (error) {
//     console.error("Decryption error:", error);
//     res.status(500).json({ error: "Error during decryption." });
//   }
// };

// module.exports = { recognizeSpeech, decryptText };

const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const forge = require("node-forge");

const keyDir = path.join(__dirname, "../keys");

if (!fs.existsSync(keyDir)) {
  fs.mkdirSync(keyDir, { recursive: true });
}

const privateKeyPath = path.join(keyDir, "private.pem");
const publicKeyPath = path.join(keyDir, "public.pem");

let publicKey, privateKey;

if (fs.existsSync(privateKeyPath) && fs.existsSync(publicKeyPath)) {
  privateKey = fs.readFileSync(privateKeyPath, "utf8");
  publicKey = fs.readFileSync(publicKeyPath, "utf8");
} else {
  const keys = forge.pki.rsa.generateKeyPair({ bits: 2048, e: 0x10001 });
  publicKey = forge.pki.publicKeyToPem(keys.publicKey);
  privateKey = forge.pki.privateKeyToPem(keys.privateKey);

  fs.writeFileSync(privateKeyPath, privateKey);
  fs.writeFileSync(publicKeyPath, publicKey);
}

const encryptWithRSA = (text, publicKey) => {
  try {
    const pubKey = forge.pki.publicKeyFromPem(publicKey);
    const encrypted = pubKey.encrypt(forge.util.encodeUtf8(text), "RSA-OAEP", {
      md: forge.md.sha256.create(),
      mgf1: {
        md: forge.md.sha256.create(),
      },
    });
    return forge.util.encode64(encrypted);
  } catch (error) {
    console.error("Error encrypting data:", error);
    return null;
  }
};

const decryptWithRSA = (encryptedText, privateKey) => {
  try {
    const privKey = forge.pki.privateKeyFromPem(privateKey);
    const decrypted = privKey.decrypt(
      forge.util.decode64(encryptedText),
      "RSA-OAEP",
      {
        md: forge.md.sha256.create(),
        mgf1: {
          md: forge.md.sha256.create(),
        },
      }
    );
    return forge.util.decodeUtf8(decrypted);
  } catch (error) {
    console.error("Error decrypting data:", error);
    return null;
  }
};

const recognizeSpeech = async (req, res) => {
  try {
    const startTime = Date.now();

    if (!req.file || !req.file.buffer) {
      return res
        .status(400)
        .json({ success: false, message: "No audio file uploaded" });
    }

    const audio = req.file.buffer;
    const webmFilePath = path.join(__dirname, `../uploads/${Date.now()}.webm`);
    fs.writeFileSync(webmFilePath, audio);
    const ttsService = req.body.module || "google";
    const pythonScriptPath = path.join(
      __dirname,
      ttsService === "test"
        ? "../scripts/process_speech2.py"
        : "../scripts/process_speech.py"
    );
    const pythonProcess = spawn("python", [
      pythonScriptPath,
      webmFilePath,
      ttsService,
    ]);

    let transcription = "";
    let errorLogs = "";

    pythonProcess.stdout.on("data", (data) => {
      const line = data.toString();
      console.log("[PYTHON STDOUT]", line); // <- Giờ thì line đã được định nghĩa
      transcription += line;
    });

    pythonProcess.stderr.on("data", (data) => {
      const errorMessage = data.toString();
      if (!errorMessage.includes("FP16 is not supported on CPU")) {
        errorLogs += errorMessage;
        console.error(`Python error: ${errorMessage}`);
      }
    });

    pythonProcess.on("close", (code) => {
      const endTime = Date.now();
      const speechProcessingTime = endTime - startTime;

      console.log(`Python process exited with code ${code}`);

      if (code === 0 && transcription.trim()) {
        const encryptionStartTime = Date.now();
        const encryptedData = encryptWithRSA(transcription.trim(), publicKey);
        const encryptionEndTime = Date.now();
        const encryptionTime = encryptionEndTime - encryptionStartTime;

        if (!encryptedData) {
          return res
            .status(500)
            .json({ success: false, message: "Encryption failed" });
        }

        return res.json({
          success: true,
          encryptedData,
          speechProcessingTime,
          encryptionTime,
        });
      } else if (code === 0 && !transcription.trim()) {
        return res.json({
          success: false,
          message: "Không nhận dạng được giọng nói",
          speechProcessingTime,
        });
      } else {
        return res.status(500).json({
          success: false,
          message: "Speech recognition failed",
          details: errorLogs || "Unknown error",
          speechProcessingTime,
        });
      }
    });
  } catch (error) {
    console.error("Error processing audio:", error);
    if (!res.headersSent) {
      res.status(500).json({
        success: false,
        message: `Error processing audio: ${error.message}`,
      });
    }
  }
};

const decryptText = async (req, res) => {
  try {
    const decryptionStartTime = Date.now();

    const { encryptedTranscript } = req.body;
    if (!encryptedTranscript) {
      return res.status(400).json({ error: "Thiếu dữ liệu để giải mã." });
    }

    const decryptedText = decryptWithRSA(encryptedTranscript, privateKey);

    const decryptionEndTime = Date.now();
    const decryptionTime = decryptionEndTime - decryptionStartTime;

    if (!decryptedText) {
      return res.status(500).json({ error: "Giải mã thất bại." });
    }

    res.json({ decryptedText, decryptionTime });
  } catch (error) {
    console.error("Lỗi giải mã:", error);
    res.status(500).json({ error: "Lỗi trong quá trình giải mã." });
  }
};

module.exports = { recognizeSpeech, decryptText };
