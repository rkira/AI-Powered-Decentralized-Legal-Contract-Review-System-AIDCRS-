const ContractReview = artifacts.require("ContractReview");

module.exports = function (deployer) {
    deployer.deploy(ContractReview);
};
