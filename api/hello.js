// 간단한 노드 서버리스 함수
module.exports = (req, res) => {
  res.status(200).json({
    message: 'Hello from Node.js serverless function!',
    timestamp: new Date().toISOString()
  });
}; 