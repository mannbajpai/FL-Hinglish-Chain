const hre = require("hardhat");

async function main() {
  const FLModelStore = await hre.ethers.getContractFactory("FLModelStore");
  const modelStore = await FLModelStore.deploy();

  await modelStore.waitForDeployment();

  console.log(`✅ Contract deployed at: ${await modelStore.getAddress()}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
