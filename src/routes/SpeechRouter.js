const express = require("express");
const multer = require("multer");
const recognitionController = require("../controllers/recognitionController");

const router = express.Router();
const upload = multer();

router.post(
  "/recognize",
  upload.single("audio"),
  recognitionController.recognizeSpeech
);
router.post("/decrypt", recognitionController.decryptText);
module.exports = router;
