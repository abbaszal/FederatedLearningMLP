import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("fig", exist_ok=True)

df_global = pd.read_csv("results_global.csv")
df_clients = pd.read_csv("results_clients.csv")

plt.figure(figsize=(10, 5))
plt.plot(df_global["Round"], df_global["Val_Accuracy"], marker="o", label="Global Val Accuracy")
plt.plot(df_global["Round"], df_global["Test_Accuracy"], marker="s", label="Global Test Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Global Validation & Test Accuracy per Round")
plt.grid(True)
plt.legend()
plt.savefig("fig/global_accuracy.png", dpi=300)
plt.savefig("fig/global_accuracy.pdf")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df_global["Round"], df_global["Val_Loss"], marker="o", label="Global Val Loss")
plt.plot(df_global["Round"], df_global["Test_Loss"], marker="s", label="Global Test Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Global Validation & Test Loss per Round")
plt.grid(True)
plt.legend()
plt.savefig("fig/global_loss.png", dpi=300)
plt.savefig("fig/global_loss.pdf")
plt.close()

plt.figure(figsize=(12, 6))
for cid in df_clients["Client_ID"].unique():
    df_c = df_clients[df_clients["Client_ID"] == cid]
    plt.plot(df_c["Round"], df_c["Accuracy"], label=f"Client {cid}")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Client Accuracy per Round")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("fig/client_accuracy.png", dpi=300)
plt.savefig("fig/client_accuracy.pdf")
plt.close()
